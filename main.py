import argparse
import json
import logging
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from aiohttp.client_exceptions import ContentTypeError
from hydrotools.nwis_client import IVDataService

from file_paths import FilePaths

logger = logging.getLogger(__name__)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# hide IVDataService warning so we can show our own
warnings.filterwarnings("ignore", message="No data was returned by the request.")


def create_crosswalk_json(hydrofabric: Path, gage_id: str, output_file: Path) -> None:
    """Create a crosswalk JSON file for a given gage ID."""
    with sqlite3.connect(hydrofabric) as con:
        sql_query = f"SELECT id FROM 'flowpath-attributes' WHERE gage = '{gage_id}'"
        result = con.execute(sql_query).fetchone()
        wb_id = result[0]
        cat_id = wb_id.replace("wb", "cat")

    data = {cat_id: {"Gage_no": gage_id}}
    with open(output_file, "w") as f:
        f.write(json.dumps(data))


def copy_and_convert_paths_to_absolute(source_file: Path, dest_file: Path) -> None:
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


def get_start_end_times(realization_path: Path) -> tuple[datetime, datetime]:
    """Get the start and end times from a realization file."""
    with open(realization_path, "r") as f:
        realization = json.loads(f.read())
    start = realization["time"]["start_time"]
    end = realization["time"]["end_time"]

    start = datetime.strptime(start, TIME_FORMAT)
    end = datetime.strptime(end, TIME_FORMAT)
    total_range = end - start
    # 2 year minimum suggested to allow for a 12 month warm up
    if total_range.days < 730:
        logger.warning(
            "Time range is less than 2 years, this may not be enough data for calibration"
        )
    return start, end


def write_usgs_data_to_csv(start: datetime, end: datetime, gage_id: str, output_file: Path) -> None:
    """Downloads the usgs observed data to csv for a given gage and time range"""
    data = pd.DataFrame()
    try:
        with IVDataService(cache_filename=FilePaths.hydrotools_cache) as service:
            data = service.get(sites=gage_id, startDT=start, endDT=end)
    except ContentTypeError:
        pass

    if data.empty:
        raise ValueError(f"Unable to find usgs observation for {gage_id} between {start} and {end}")

    data = data[["value_time", "value"]]
    data.columns = ["value_date", "obs_flow"]
    # usgs data is in ft3/s, ngen-cal converts to m3/s without checking so LEAVE IT AS ft3/s
    data.to_csv(output_file, index=False)


def write_ngen_cal_config(
    data_folder: FilePaths, gage_id: str, start: datetime, end: datetime
) -> None:
    total_range = start - end
    # warm up is half the range, capped at 365 days
    warm_up = timedelta(days=(total_range.days / 2))
    if warm_up.days < 365:
        warm_up = timedelta(days=365)
    # evaluation starts at the end of the warm up period
    # Validation not currently working so just set the values the same as eval
    evaluation_start = validation_start = start + warm_up
    evaluation_end = validation_end = end

    # ends after half the remaining time
    # evaluation_end = end - ((total_range - warm_up) / 2)
    # # validation starts at the end of the evaluation period
    # validation_start = evaluation_end
    # validation_end = end

    with open(FilePaths.template_ngiab_cal_conf, "r") as f:
        template = f.read()

    with open(data_folder.calibration_config, "w") as file:
        file.write(
            template.format(
                subset_hydrofabric=data_folder.geopackage_path.name,
                evaluation_start=evaluation_start.strftime(TIME_FORMAT),
                evaluation_stop=evaluation_end.strftime(TIME_FORMAT),
                valid_start_time=start.strftime(TIME_FORMAT),
                valid_end_time=end.strftime(TIME_FORMAT),
                valid_eval_start_time=validation_start.strftime(TIME_FORMAT),
                valid_eval_end_time=validation_end.strftime(TIME_FORMAT),
                full_eval_start_time=start.strftime(TIME_FORMAT),
                full_eval_end_time=end.strftime(TIME_FORMAT),
                gage_id=gage_id,
            )
        )


def get_gages_from_hydrofabric(hydrofabric: Path) -> list[str]:
    with sqlite3.connect(hydrofabric) as conn:
        sql = "select gage from 'flowpath-attributes' where gage is not NULL"
        return [row[0] for row in conn.execute(sql).fetchall()]


def pick_gage_to_calibrate(hydrofabric: Path) -> str:
    gages = get_gages_from_hydrofabric(hydrofabric)
    if len(gages) == 1:
        return gages[0]
    else:
        return input(f"Select a gage to calibrate from {gages}: ")


def create_calibration_config(data_folder: Path, gage_id: str) -> None:
    # first pass at this so I'm probably not using ngen-cal properly
    # for now keep it simple and only allow single gage lumped calibration

    # This initialization also checks all the files we need exist
    files = FilePaths(data_folder)
    if not gage_id:
        gage_id = pick_gage_to_calibrate(files.geopackage_path)

    all_gages = get_gages_from_hydrofabric(files.geopackage_path)
    if gage_id not in all_gages:
        raise ValueError(
            f"Gage {gage_id} not in {files.geopackage_path}, avaiable options are {all_gages}"
        )

    start, end = get_start_end_times(files.template_realization)
    write_usgs_data_to_csv(start, end, gage_id, files.observed_discharge)
    create_crosswalk_json(files.geopackage_path, gage_id, files.crosswalk)
    # copy the ngen realization and troute config files into the calibration folder
    # convert the relative paths to absolute for ngiab_cal compatibility
    copy_and_convert_paths_to_absolute(files.template_realization, files.calibration_realization)
    copy_and_convert_paths_to_absolute(files.template_troute, files.calibration_troute)

    # create the dates for the ngen-cal config
    write_ngen_cal_config(files, gage_id, start, end)

    print("This is still experimental, run the following command to start calibration:")
    print(f'docker run -it -v "{files.data_folder}:/ngen/ngen/data" joshcu/ngiab-cal')


# def generate_best_realization(calibration_dir: Path) -> None:
#     # this relies too much on the folder structure of this, ngiab, and ngen-cal
#     output_realization = calibration_dir / "../config/" / "best_realization.json"
#     source_realization = calibration_dir / "../config/" / "realization.json"

#     # read the calibration config to figure out what parameters belong to which model
#     calibration_config = calibration_dir / "ngen_cal_conf.yaml"
#     with open(calibration_config, "r") as f:
#         config = f.read()
#     config = yaml.safe_load(config)

#     # dictionary of model names to their parameters
#     models = config["model"]["params"]

#     parameters = {}

#     # the default config uses some kind of linked yaml object that won't load
#     for model_name in models:
#         model = config[model_name]
#         parameters[model_name] = []
#         for param in model:
#             parameters[model_name].append(param["name"])

#     # now we have dictionary of model names to their parameters
#     with open(source_realization, "r") as f:
#         realization = json.loads(f.read())

#     # now to get the best parameters
#     objective_metric = config["model"]["eval_params"]["objective"]

#     # get all the worker outputs
#     # glob calibration_dir/Output/*/ngen_*_worker/*_metrics_iteration.csv
#     # look for the column with the objective metric and find the row with the best value
#     # then open *_params_iteration.csv in the same folder to get the parameters
#     # then update the realization.json file with the new parameters
#     metric_files = calibration_dir.glob("Output/*/ngen_*_worker/*_metrics_iteration.csv")
#     current_best = None
#     for file in metric_files:
#         metric_df = pd.read_csv(file)
#         # check the headers to match the case of the objective metric
#         if objective_metric.lower() in metric_df.columns:
#             objective_metric = objective_metric.lower()
#         if objective_metric.upper() in metric_df.columns:
#             objective_metric = objective_metric.upper()
#         # register the best value on the first run
#         if current_best is None:
#             current_best = metric_df[objective_metric].max()

#         # if the current workers best worse than the current best, skip it
#         if metric_df[objective_metric].max() < current_best:
#             continue
#         best_row = metric_df[metric_df[objective_metric] == metric_df[objective_metric].max()]
#         best_iteration = best_row["iteration"]
#         worker_dir = file.parent
#         iteration_file = list(worker_dir.glob("*_params_iteration.csv"))[0]
#         params_df = pd.read_csv(iteration_file)
#         params_df = params_df.set_index("iteration")
#         best_params = pd.read_csv(iteration_file).iloc[best_iteration]

#     realization_modules = realization["global"]["formulations"][0]["params"]["modules"]

#     # loop over the models used in the realization
#     for module in realization_modules:
#         module_name = module["params"]["model_type_name"]
#         if module_name in parameters:
#             model_params = {}
#             for parameter_name in parameters[module_name]:
#                 if parameter_name in best_params:
#                     model_params[parameter_name] = best_params[parameter_name].values[0]
#             module["params"]["model_params"] = model_params

#     with open(output_realization, "w") as f:
#         f.write(json.dumps(realization))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a calibration config for ngen-cal")
    parser.add_argument(
        "data_folder",
        type=Path,
        help="Path to the folder you wish to calibrate",
    )
    parser.add_argument("-g", "--gage", type=str, help="Gage ID to use for calibration")
    args = parser.parse_args()
    create_calibration_config(args.data_folder, args.gage)
