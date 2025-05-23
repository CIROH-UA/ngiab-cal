import json
import logging
import sqlite3
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from aiohttp.client_exceptions import ContentTypeError
from hydrotools.nwis_client import IVDataService

from ngiab_cal.arguments import (
    CALIBRATION_VALIDATION_RATIO,
    ITERATIONS_DEFAULT,
    WARMUP_DEFAULT,
    get_arg_parser,
)
from ngiab_cal.custom_logging import set_log_level, setup_logging
from ngiab_cal.file_paths import FilePaths, validate_calibration_files, validate_run_files

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DOCKER_IMAGE_NAME = "awiciroh/ngiab-cal:devcon25"


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
                line = line.replace("/ngen/ngen/data/outputs/ngen", ".")
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
        logging.warning(
            "Time range is less than 2 years, this may not be enough data for calibration"
        )
    return start, end


def write_usgs_data_to_csv(start: datetime, end: datetime, gage_id: str, output_file: Path) -> None:
    """Downloads the usgs observed data to csv for a given gage and time range"""
    logging.info(f"Downloading USGS data for {gage_id} between {start} and {end}")
    data = pd.DataFrame()
    try:
        with IVDataService(cache_filename=FilePaths.hydrotools_cache) as service:
            data = service.get(sites=gage_id, startDT=start, endDT=end)
    except ContentTypeError:
        pass

    if data.empty:
        raise ValueError(f"Unable to find usgs observation for {gage_id} between {start} and {end}")

    data = data.filter(["value_time", "value"])
    data.columns = ["value_date", "obs_flow"]
    # usgs data is in ft3/s, ngen-cal converts to m3/s without checking so LEAVE IT AS ft3/s
    data.to_csv(output_file, index=False)


def write_ngen_cal_config(
    data_folder: FilePaths,
    gage_id: str,
    start: datetime,
    end: datetime,
    iterations: int,
    warm_up: int,
    calibration_ratio: float,
) -> None:
    logging.info("Writing ngiab-cal configuration")
    total_range = abs(start - end)
    # warm up is half the range, capped at 365 days
    warm_up_period = timedelta(days=warm_up)
    if warm_up_period > total_range:
        logging.error(
            f"Warm up period {warm_up_period} days is longer that the total range to be simulated {total_range} "
        )
    # Validation not currently working so just set the values the same as eval
    # round to the nearest day so we don't get strange intervals ngen isn't expecting
    calibration_days = ((total_range - warm_up_period) * calibration_ratio).days
    calibration_td = timedelta(days=calibration_days)
    evaluation_start = start + warm_up_period
    # ends after half the remaining time
    if calibration_ratio == 0:
        evaluation_end = end
    else:
        evaluation_end = evaluation_start + calibration_td
    # validation starts at the end of the evaluation period
    validation_start = evaluation_end
    validation_end = end

    # debug all the times
    logging.debug("start {}".format(start))
    logging.debug("end {}".format(end))
    logging.debug("Total range: {}".format(total_range))
    logging.debug("Warm up: {}".format(warm_up))
    logging.debug("Calibration Days: {}".format(calibration_days))
    logging.debug("Validation Days: {}".format(abs(validation_start - validation_end).days))
    logging.debug("Evaluation start: {}".format(evaluation_start))
    logging.debug("Evaluation end: {}".format(evaluation_end))
    logging.debug("Validation start: {}".format(validation_start))
    logging.debug("Validation end: {}".format(validation_end))

    plot_frequency = min(50, iterations)

    with open(FilePaths.template_ngiab_cal_conf, "r") as f:
        template = f.read()

    with open(data_folder.calibration_config, "w") as file:
        file.write(
            template.format(
                iterations=iterations,
                plot_frequency=plot_frequency,
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


def create_calibration_config(
    data_folder: Path,
    gage_id: str,
    iterations: int = ITERATIONS_DEFAULT,
    warmup_days: int = WARMUP_DEFAULT,
    calib_ratio: float = CALIBRATION_VALIDATION_RATIO,
) -> None:
    # first pass at this so I'm probably not using ngen-cal properly
    # for now keep it simple and only allow single gage lumped calibration

    logging.info("Validating input files")
    # This initialization also checks all the files we need exist
    files = FilePaths(data_folder)

    files.calibration_folder.mkdir(exist_ok=True)

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
    write_ngen_cal_config(files, gage_id, start, end, iterations, warmup_days, calib_ratio)


def print_run_command(folder_to_run: Path) -> None:
    logging.info("This is still experimental, run the following command to start calibration:")
    logging.info(
        f'docker run -it -v "{folder_to_run.resolve()}:/ngen/ngen/data" --user $(id -u):$(id -g) {DOCKER_IMAGE_NAME}'
    )


def run_calibration(folder_to_run: Path) -> None:
    try:
        subprocess.run(f"docker pull {DOCKER_IMAGE_NAME}", shell=True)
    except:
        logging.error("Docker is not running, please start Docker and try again.")
    logging.warning("Beginning calibration...")
    try:
        command = f'docker run --rm -it -v "{str(folder_to_run.resolve())}:/ngen/ngen/data" --user $(id -u):$(id -g) {DOCKER_IMAGE_NAME} /calibration/run.sh'
        subprocess.run(command, shell=True)
        logging.info("Calibration complete.")
    except:
        logging.error("Calibration failed.")


def copy_best_params(realization: Path, calibrated_realization: Path) -> None:
    if not realization.exists() or not calibrated_realization.exists():
        logging.error(
            f"Realization path {realization} or calibrated realization path {calibrated_realization} does not exist."
        )
    logging.debug(
        f"Extracting model parameters from calibrated realization: {calibrated_realization}"
    )
    with open(realization) as f:
        uncalibrated = json.load(f)
    with open(realization.with_suffix("._old"), "w") as f:
        json.dump(uncalibrated, f, indent=4)
    with open(calibrated_realization) as f:
        calibrated = json.load(f)

    # this assumes that the calibration was done in ngiab
    old_modules = uncalibrated["global"]["formulations"][0]["params"]["modules"]
    new_modules = calibrated["global"]["formulations"][0]["params"]["modules"]

    for old_module in old_modules:
        for new_module in new_modules:
            if old_module["params"]["model_type_name"] == new_module["params"]["model_type_name"]:
                old_module["params"]["model_params"] = new_module["params"]["model_params"]

    logging.info(
        f"Copying model parameters from calibrated realization {calibrated_realization} into {realization}"
    )
    with open(realization, "w") as f:
        json.dump(uncalibrated, f, indent=4)


def main():
    setup_logging()
    args = get_arg_parser().parse_args()

    if args.debug:
        set_log_level(logging.DEBUG)
    paths = FilePaths(args.data_folder)

    data_folder_valid = validate_run_files(paths, log_level=logging.ERROR)

    if not data_folder_valid:
        return

    logging.debug(f"Searching for calibration files in {paths.calibration_folder}")
    config_valid = validate_calibration_files(paths, log_level=logging.NOTSET)
    if not config_valid and paths.calibration_folder.exists():
        # e.g. if there are missing config files and this isn't the first run
        # warn that there are files missing
        validate_calibration_files(paths, log_level=logging.WARN)

    if config_valid and not args.force:
        logging.warning(
            "Existing calibration configuration found, use -f or --force to overwrite existing calibration settings"
        )
    # drop the gage- syntax used in other tools
    if args.gage:
        args.gage = args.gage.split("-")[-1]
    if not config_valid or args.force:
        create_calibration_config(
            args.data_folder, args.gage, args.iterations, args.warmup, args.calibration_ratio
        )

    if args.run:
        logging.info(f"Starting calibration run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        run_calibration(args.data_folder)

    config_valid = validate_calibration_files(paths, log_level=logging.NOTSET)
    if config_valid:
        print_run_command(args.data_folder)

    if paths.calibrated_realization.exists():
        logging.info(f"Calibrated realization found: {paths.calibrated_realization}")
        copy_best_params(paths.template_realization, paths.calibrated_realization)


if __name__ == "__main__":
    sys.exit(main())
