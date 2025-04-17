import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def search_for_file(expected_file: Path, search_glob: str) -> Path:
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
            return Path(files_found[0])
        case _:
            raise FileExistsError(
                f"too many files matching {search_glob}, found {num_files} in {expected_file.parent}"
            )


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
        if not data_folder.exists():
            raise FileNotFoundError(f"Unable to find {data_folder}")
        self.data_folder = data_folder
        validate_input_folder(self, skip_calibration_folder=True)

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
        return search_for_file(expected_file, search_glob="**/*.gpkg")

    @property
    def best_realization(self) -> Path:
        return self.config_folder / "calibrated.json"

    @property
    def template_realization(self) -> Path:
        return search_for_file(self.config_folder / "realization.json", "**/real*.json")

    @property
    def template_troute(self) -> Path:
        return search_for_file(self.config_folder / "troute.yaml", "**/*rout*.yaml")

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


def validate_input_folder(data_folder: FilePaths, skip_calibration_folder: bool = True) -> bool:
    """
    Checks all the file and folders required for calibration are present.
    Loops over all properties of an object that return a path and checks that they exist.
    """
    skip_attrs = ["best_realization"]
    data_folder.calibration_folder.mkdir(exist_ok=True)
    missing_paths = list()
    for attr_name in dir(data_folder):
        # Skip private attributes and methods
        if attr_name.startswith("_") or attr_name in skip_attrs:
            continue

        try:
            # Check if it's a property that returns a path-like object
            path = getattr(data_folder, attr_name)
            if isinstance(path, (Path)) and not callable(path):
                if skip_calibration_folder and path.is_relative_to(data_folder.calibration_folder):
                    continue
                if not path.exists():
                    raise FileNotFoundError(f"unable to locate {attr_name} at {path}")
        except Exception as e:
            # Handle any exceptions (e.g., if a property raises an error)
            missing_paths.append(f"{str(e)}")
    if len(missing_paths) == 0:
        return True
    else:
        for missing_path in missing_paths:
            logger.error(missing_path)
        return False


# if __name__ == "__main__":
#     # test = file_paths(Path())
#     test = file_paths(Path("/mnt/raid0/cal_testing/gage-10154200"))
