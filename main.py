import argparse
from data_processing.file_paths import file_paths
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import logging
from hydrotools.nwis_client import IVDataService
import yaml
import pandas as pd

logger = logging.getLogger(__name__)


def create_crosswalk_json(hydrofabric_path: Path, gage_id: str, output_file: Path) -> None:
    if not hydrofabric_path.exists():
        raise FileNotFoundError(
            f"Hydrofabric path {hydrofabric_path} does not exist, have you run the subset command?"
        )

    with sqlite3.connect(hydrofabric_path) as con:
        sql_query = f"SELECT id FROM 'flowpath-attributes' WHERE gage = '{gage_id}'"
        result = con.execute(sql_query).fetchall()
        if len(result) == 0:
            logger.critical(f"Gage ID {gage_id} is not associated with any waterbodies")
            raise IndexError(f"Could not find a waterbody for gage {gage_id}")
        if len(result) > 1:
            logger.critical(f"Gage ID {gage_id} is associated with multiple waterbodies")
            raise IndexError(f"Could not find a unique waterbody for gage {gage_id}")

        wb_id = result[0][0]
        cat_id = wb_id.replace("wb", "cat")

    data = {cat_id: {"Gage_no": gage_id}}
    with open(output_file, "w") as f:
        f.write(json.dumps(data))


def convert_paths_to_absolute(source_file: Path, dest_file: Path) -> None:
    # a bit dodgy but removeable once ngiab-cal is updated
    with open(source_file, "r") as f:
        with open(dest_file, "w") as out:
            for line in f:
                line = line.replace("./", "/ngen/ngen/data/")
                line = line.replace("/ngen/ngen/data/outputs/ngen/", ".")
                line = line.replace("outputs/troute/", ".")
                # ngiab-cal takes troute yaml as an input but doesn't replace this value
                line = line.replace(
                    "/ngen/ngen/data/config/troute.yaml", "/ngen/ngen/data/calibration/troute.yaml"
                )
                if "lakeout_output" in line:
                    continue
                if "lite_restart" in line:
                    continue
                out.write(line)


def setup_ngen_troute_config(paths: file_paths) -> None:
    realization_path = paths.config_dir / "realization.json"
    troute_path = paths.config_dir / "troute.yaml"
    # copy over the realization.json file and troute.yaml file
    if not realization_path.exists():
        raise FileNotFoundError(
            f"Realization file {realization_path} does not exist, have you run realization generation?"
        )
    if not troute_path.exists():
        raise FileNotFoundError(
            f"Troute config file {troute_path} does not exist, have you run the subset command?"
        )
    convert_paths_to_absolute(troute_path, paths.calibration_dir / "troute.yaml")
    convert_paths_to_absolute(realization_path, paths.calibration_dir / "realization.json")


def get_start_end_times(realization_path: Path) -> tuple[str, str]:
    with open(realization_path, "r") as f:
        realization = json.loads(f.read())
    start = realization["time"]["start_time"]
    end = realization["time"]["end_time"]
    return start, end


def write_usgs_data_to_csv(start: str, end: str, gageid: str, output_file: Path) -> None:
    service = IVDataService(cache_filename="~/.ngiab/hydrotools_cache.sqlite")
    data = service.get(sites=gageid, startDT=start, endDT=end)
    unit = data["measurement_unit"][0]
    data = data[["value_time", "value"]]
    data.columns = ["value_date", "obs_flow"]
    # usgs data is in ft3/s, ngen-cal converts to m3/s without checking so LEAVE IT AS ft3/s
    data.to_csv(output_file, index=False)


def create_ngen_cal_config(output_dir: Path, gage_id: str, start_str: str, end_str: str) -> None:
    time_format = "%Y-%m-%d %H:%M:%S"
    start = datetime.strptime(start_str, time_format)
    end = datetime.strptime(end_str, time_format)
    total_range = end - start
    # 2 year minimum suggested, 12 month warm up, then 50/50 split between calibration and validation
    # if more than 2 years, 12 month warm up and the rest is split 50/50
    if total_range.days < 730:
        logger.warning(
            "Calibration period is less than 2 years, this may not be enough data for calibration"
        )
    # warm up is half the range, capped at 365 days
    warm_up = timedelta(days=(total_range.days / 2))
    if warm_up.days < 365:
        warm_up = timedelta(days=365)
    # evaluation starts at the end of the warm up period
    evaluation_start = start + warm_up
    # ends after half the remaining time
    evaluation_end = end - ((total_range - warm_up) / 2)
    # validation starts at the end of the evaluation period
    validation_start = evaluation_end
    validation_end = end

    with open(file_paths.template_calibration_config, "r") as f:
        template = f.read()

    paths = file_paths(output_dir=output_dir)

    with open(paths.calibration_dir / "ngen_cal_conf.yaml", "w") as file:
        file.write(
            template.format(
                subset_hydrofabric=paths.geopackage_path.name,
                evaluation_start=evaluation_start.strftime(time_format),
                evaluation_stop=evaluation_end.strftime(time_format),
                valid_start_time=start.strftime(time_format),
                valid_end_time=end.strftime(time_format),
                valid_eval_start_time=validation_start.strftime(time_format),
                valid_eval_end_time=validation_end.strftime(time_format),
                full_eval_start_time=start.strftime(time_format),
                full_eval_end_time=end.strftime(time_format),
                gage_id=gage_id,
            )
        )


def create_calibration_config(calibration_dir: Path, gage_id: str) -> None:
    # first pass at this so I'm probably not using ngen-cal properly
    # for now keep it simple and only allow single gage lumped calibration
    gage_id = gage_id.split("-")[-1]
    # get the output directory
    paths = file_paths(output_dir=calibration_dir)
    if not paths.calibration_dir.exists():
        paths.calibration_dir.mkdir(parents=True)
    # create crosswalk.json
    create_crosswalk_json(paths.geopackage_path, gage_id, paths.calibration_dir / "crosswalk.json")
    # copy over the realization.json file and troute.yaml file
    # update the paths to be non relative
    setup_ngen_troute_config(paths)
    # get start and end times from realization.json
    start, end = get_start_end_times(paths.config_dir / "realization.json")
    # get the observed flow for the gage in for the time period
    write_usgs_data_to_csv(start, end, gage_id, paths.calibration_dir / "obs_hourly_discharge.csv")
    # create the dates for the ngen-cal config
    create_ngen_cal_config(paths.output_dir, gage_id, start, end)


def generate_best_realization(calibration_dir: Path) -> None:
    # this relies too much on the folder structure of this, ngiab, and ngen-cal
    output_realization = calibration_dir / "../config/" / "best_realization.json"
    source_realization = calibration_dir / "../config/" / "realization.json"

    # read the calibration config to figure out what parameters belong to which model
    calibration_config = calibration_dir / "ngen_cal_conf.yaml"
    with open(calibration_config, "r") as f:
        config = f.read()
    config = yaml.safe_load(config)

    # dictionary of model names to their parameters
    models = config["model"]["params"]

    parameters = {}

    # the default config uses some kind of linked yaml object that won't load
    for model_name in models:
        model = config[model_name]
        parameters[model_name] = []
        for param in model:
            parameters[model_name].append(param["name"])

    # now we have dictionary of model names to their parameters
    with open(source_realization, "r") as f:
        realization = json.loads(f.read())

    # now to get the best parameters
    objective_metric = config["model"]["eval_params"]["objective"]

    # get all the worker outputs
    # glob calibration_dir/Output/*/ngen_*_worker/*_metrics_iteration.csv
    # look for the column with the objective metric and find the row with the best value
    # then open *_params_iteration.csv in the same folder to get the parameters
    # then update the realization.json file with the new parameters
    metric_files = calibration_dir.glob("Output/*/ngen_*_worker/*_metrics_iteration.csv")
    current_best = None
    for file in metric_files:
        metric_df = pd.read_csv(file)
        # check the headers to match the case of the objective metric
        if objective_metric.lower() in metric_df.columns:
            objective_metric = objective_metric.lower()
        if objective_metric.upper() in metric_df.columns:
            objective_metric = objective_metric.upper()
        # register the best value on the first run
        if current_best is None:
            current_best = metric_df[objective_metric].max()

        # if the current workers best worse than the current best, skip it
        if metric_df[objective_metric].max() < current_best:
            continue
        best_row = metric_df[metric_df[objective_metric] == metric_df[objective_metric].max()]
        best_iteration = best_row["iteration"]
        worker_dir = file.parent
        iteration_file = list(worker_dir.glob("*_params_iteration.csv"))[0]
        params_df = pd.read_csv(iteration_file)
        params_df = params_df.set_index("iteration")
        best_params = pd.read_csv(iteration_file).iloc[best_iteration]

    realization_modules = realization["global"]["formulations"][0]["params"]["modules"]

    # loop over the models used in the realization
    for module in realization_modules:
        module_name = module["params"]["model_type_name"]
        if module_name in parameters:
            model_params = {}
            for parameter_name in parameters[module_name]:
                if parameter_name in best_params:
                    model_params[parameter_name] = best_params[parameter_name].values[0]
            module["params"]["model_params"] = model_params

    with open(output_realization, "w") as f:
        f.write(json.dumps(realization))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a calibration config for ngen-cal")
    parser.add_argument(
        "-d",
        "--calibration_dir",
        type=Path,
        help="Directory to create the calibration config in",
    )
    parser.add_argument("-g", "--gage", type=str, help="Gage ID to use for calibration")
    args = parser.parse_args()
    create_calibration_config(args.calibration_dir, args.gage)
