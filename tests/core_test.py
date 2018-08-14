from unittest.mock import Mock

import basilisp.lang.runtime as runtime
from basilisp.main import init

init()

# Cache the initial state of the `print_generated_python` flag.
__PRINT_GENERATED_PYTHON_FN = runtime.print_generated_python


def setup_module(module):
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    runtime.print_generated_python = Mock(return_value=False)


def teardown_module(module):
    """Restore the `print_generated_python` flag after we finish running tests."""
    runtime.print_generated_python = __PRINT_GENERATED_PYTHON_FN


import basilisp.core


def test_first():
    assert None is basilisp.core.first(None)
