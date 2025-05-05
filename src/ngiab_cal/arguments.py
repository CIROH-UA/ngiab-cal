from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

ITERATIONS_DEFAULT = 100
WARMUP_DEFAULT = 365
CALIBRATION_VALIDATION_RATIO = 0.5


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create a calibration config for ngen-cal",
        epilog="""
                 Default calibration settings on a 5 year period
        |   year 1   |   year 2   |   year 3   |   year 4   |   year 5   |
        |<- warmup ->|<-     calibration     ->|      no simulation      |
        |<-             warmup               ->|<-      validation     ->|
        """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "data_folder",
        type=Path,
        help="Path to the folder you wish to calibrate",
    )
    parser.add_argument("-g", "--gage", type=str, help="Gage ID to use for calibration")
    parser.add_argument(
        "-f", "--force", help="Overwrite existing configuration", action="store_true"
    )
    parser.add_argument(
        "--run",
        help="Try to automatically run the calibration, this may be unstable",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        help=f"Default:{ITERATIONS_DEFAULT} number of iterations to calibrate for\n\n",
        type=int,
        default=ITERATIONS_DEFAULT,
    )
    parser.add_argument("--debug", help="enable debug logging", action="store_true")
    parser.add_argument(
        "-w",
        "--warmup",
        help=f"Default:{WARMUP_DEFAULT}\nNumber of days at the beginning of the simulation\n to exclude from calibration objective metric calculation",
        type=int,
        default=WARMUP_DEFAULT,
    )
    parser.add_argument(
        "--calibration_ratio",
        "--cr",
        type=float,
        help=f"Default:{CALIBRATION_VALIDATION_RATIO}\nHow to split time after warmup into calibration and validation.\n1 == 100%% calibration, 0 == 100%% validation, 0.8 == 80%% calibration 20%% validation",
        default=CALIBRATION_VALIDATION_RATIO,
    )

    return parser
