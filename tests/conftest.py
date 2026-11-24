import base64
import os
import pickle
import subprocess
import sys
from typing import Callable

import pytest

pytest_plugins = ["pytester"]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Marks tests as slow")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run slow tests")


@pytest.fixture(autouse=True)
def skip_slow(request):
    if request.node.get_closest_marker("slow") and not request.config.getoption(
        "--run-slow"
    ):
        pytest.skip("Skipping slow test. Use --run-slow to enable.")


@pytest.fixture
def pickle_cross_process():
    """Return a fixture which runs two subprocesses (invocations of the current
    process executable) with code which generates a serialized pickled output in one
    subprocess and then deserializes that same output in a separate process.

    The intent is to check for any values which might store stateful hashes computed
    based on the current PYTHONHASHSEED value which are not valid in other processes.
    """

    def _call(fn, *args, seed: int):
        payload = base64.b64encode(pickle.dumps((fn, args)))
        runner = (
            "import base64, pickle, sys; "
            "fn, args = pickle.loads(base64.b64decode(sys.stdin.buffer.readline())); "
            "out = fn(*args); "
            "sys.stdout.buffer.write(base64.b64encode(pickle.dumps(out)))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=payload,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        return pickle.loads(base64.b64decode(proc.stdout))

    def _pickle_cross_process(producer: Callable, consumer: Callable):
        blob = _call(producer, seed=1)
        return _call(consumer, blob, seed=2)

    return _pickle_cross_process
