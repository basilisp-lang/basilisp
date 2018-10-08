import importlib
import logging
import os

import basilisp.importer as importer
import basilisp.lang.runtime as runtime


def _get_default_level() -> str:
    """Get the default logging level for Basilisp."""
    return os.getenv('BASILISP_LOGGING_LEVEL', 'WARNING')


def _get_default_handler(level: str, formatter: logging.Formatter) -> logging.Handler:
    """Get the default logging handler for Basilisp."""
    handler: logging.Handler = logging.NullHandler()
    if os.getenv('BASILISP_USE_DEV_LOGGER') == 'true':
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


_DEFAULT_FORMAT = '%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s'
_DEFAULT_FORMATTER = logging.Formatter(_DEFAULT_FORMAT)
_DEFAULT_LEVEL = _get_default_level()
_DEFAULT_HANDLER = _get_default_handler(_DEFAULT_LEVEL, _DEFAULT_FORMATTER)

logger = logging.getLogger('basilisp')
logger.setLevel(_DEFAULT_LEVEL)
logger.addHandler(_DEFAULT_HANDLER)


def init():
    """Initialize the runtime environment for evaluation."""
    runtime.init_ns_var()
    runtime.bootstrap()
    importer.hook_imports()
    importlib.import_module('basilisp.core')
