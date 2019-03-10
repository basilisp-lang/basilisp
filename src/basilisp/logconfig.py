import logging
import os


def get_level() -> str:
    """Get the default logging level for Basilisp."""
    return os.getenv("BASILISP_LOGGING_LEVEL", "WARNING")


def get_handler(level: str, fmt: str) -> logging.Handler:
    """Get the default logging handler for Basilisp."""
    handler: logging.Handler = logging.NullHandler()
    if os.getenv("BASILISP_USE_DEV_LOGGER") == "true":
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)
    return handler


TRACE = 5

logging.addLevelName(TRACE, "TRACE")

DEFAULT_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)
DEFAULT_LEVEL = get_level()
DEFAULT_HANDLER = get_handler(DEFAULT_LEVEL, DEFAULT_FORMAT)
