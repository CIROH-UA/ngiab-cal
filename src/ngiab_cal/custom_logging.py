import logging

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{Fore.BLUE}{message}{Style.RESET_ALL}"
        if record.levelno == logging.WARNING:
            return f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        if record.levelno == logging.CRITICAL or record.levelno == logging.ERROR:
            return f"{Fore.RED}{message}{Style.RESET_ALL}"
        if record.name == "root":  # Only color info messages from this script green
            return f"{Fore.GREEN}{message.split(' - ')[0]}{Style.RESET_ALL} - {message.split(' - ')[1]}"
        return message


def setup_logging() -> None:
    """Set up logging configuration with green formatting."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # set aiohttp to info because it's NOTSET by default
    # an extremely helpful tool https://pypi.org/project/logging-tree/
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    logging.getLogger("aiohttp_client_cache").setLevel(logging.INFO)
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("concurrent").setLevel(logging.INFO)


def set_log_level(log_level: int) -> None:
    logger = logging.getLogger("root")
    logger.setLevel(log_level)


def set_logging_to_critical_only() -> None:
    """Set logging to CRITICAL level only."""
    logging.getLogger().setLevel(logging.CRITICAL)
    # Explicitly set Dask's logger to CRITICAL level
    logging.getLogger("distributed").setLevel(logging.CRITICAL)
