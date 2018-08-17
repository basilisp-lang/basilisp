from unittest.mock import Mock

import pytest

import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.vector as vec
from basilisp.lang.exception import ExceptionInfo
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


import basilisp.core as core


def test_first():
    assert None is core.first(None)


def test_ex_info():
    with pytest.raises(ExceptionInfo):
        raise core.ex_info("This is just an exception", lmap.m())


def test_last():
    assert None is core.last(llist.List.empty())
    assert 1 == core.last(llist.l(1))
    assert 2 == core.last(llist.l(1, 2))
    assert 3 == core.last(llist.l(1, 2, 3))

    assert None is core.last(vec.Vector.empty())
    assert 1 == core.last(vec.v(1))
    assert 2 == core.last(vec.v(1, 2))
    assert 3 == core.last(vec.v(1, 2, 3))
