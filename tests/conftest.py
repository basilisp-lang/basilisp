import pytest

pytest_plugins = ["pytester"]

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Marks tests as slow")

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run slow tests")

@pytest.fixture(autouse=True)
def skip_slow(request):
    if request.node.get_closest_marker("slow") and not request.config.getoption("--run-slow"):
        pytest.skip("Skipping slow test. Use --run-slow to enable.")
