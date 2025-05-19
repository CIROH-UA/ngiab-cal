import glob
import logging
from pathlib import Path


def _search_for_file(expected_file: Path, search_glob: str) -> Path:
    if (expected_file).exists():
        return expected_file
    files_found = glob.glob(search_glob, root_dir=expected_file.parent, recursive=True)
    num_files = len(files_found)
    match num_files:
        case 0:
            raise FileNotFoundError(
                f"unable to find any files matching {search_glob} in {expected_file.parent}"
            )
        case 1:
            return expected_file.parent / Path(files_found[0])
        case _:
            raise FileExistsError(
                f"too many files matching {search_glob}, found {num_files} in {expected_file.parent}"
            )


def _validate_files(paths: list[Path], log_level: int = logging.WARN):
    missing_paths = []
    for path in paths:
        try:
            if not path.exists():
                raise FileNotFoundError(f"unable to locate {path}")
        except FileNotFoundError as e:
            missing_paths.append(f"{str(e)}")
        except FileExistsError as e:
            missing_paths.append(f"{str(e)}")

    if len(missing_paths) == 0:
        return True
    else:
        if log_level == logging.NOTSET:
            return False
        for missing_path in missing_paths:
            logging.log(log_level, missing_path)
        return False


class FilePaths:
    """
    This class contains all of the file paths used in the calibration workflow
    workflow.
    """

    template_ngiab_cal_conf = Path(__file__).parent / "ngiab_cal_template.yaml"
    hydrotools_cache = Path("~/.ngiab/hydrotools_cache.sqlite").expanduser()

    def __init__(self, data_folder: Path):
        """
        Initialize the file_paths class with a path to the folder you want to calibrate.
        Args:
            folder_name (str): Water body ID.
            output_folder (Path): Path to the folder you want to output to
        """
        data_folder = data_folder.resolve()
        if not data_folder.exists():
            raise FileNotFoundError(f"Unable to find {data_folder}")
        self.data_folder = data_folder
        # validate_input_folder(self, skip_calibration_folder=True, log_level=logging.ERROR)

    @property
    def calibration_folder(self) -> Path:
        return self.data_folder / "calibration"

    @property
    def config_folder(self) -> Path:
        return self.data_folder / "config"

    @property
    def forcings_folder(self) -> Path:
        return self.data_folder / "forcings"

    @property
    def geopackage_path(self) -> Path:
        expected_file = self.config_folder / f"{self.data_folder.stem}_subset.gpkg"
        return _search_for_file(expected_file, search_glob="**/*.gpkg")

    @property
    def template_realization(self) -> Path:
        return _search_for_file(self.config_folder / "realization.json", "**/real*.json")

    @property
    def template_troute(self) -> Path:
        return _search_for_file(self.config_folder / "troute.yaml", "**/*rout*.yaml")

    @property
    def calibration_realization(self) -> Path:
        return self.calibration_folder / "realization.json"

    @property
    def calibration_troute(self) -> Path:
        return self.calibration_folder / "troute.yaml"

    @property
    def calibration_config(self) -> Path:
        return self.calibration_folder / "ngen_cal_conf.yaml"

    @property
    def crosswalk(self) -> Path:
        return self.calibration_folder / "crosswalk.json"

    @property
    def observed_discharge(self) -> Path:
        return self.calibration_folder / "obs_hourly_discharge.csv"

    @property
    def calibrated_realization(self) -> Path:
        return self.calibration_folder / "Output" / "Validation_Run" / "realization.json"


def validate_run_files(self, log_level: int = logging.WARN) -> bool:
    required_run_files = [
        self.config_folder,
        self.forcings_folder,
        self.geopackage_path,
        self.template_realization,
        self.template_troute,
    ]
    return _validate_files(required_run_files, log_level)


def validate_calibration_files(self, log_level: int = logging.WARN) -> bool:
    required_calibration_files = [
        self.calibration_realization,
        self.calibration_troute,
        self.calibration_config,
        self.crosswalk,
        self.observed_discharge,
    ]
    return _validate_files(required_calibration_files, log_level)
