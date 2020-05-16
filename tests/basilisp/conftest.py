import sys

import pytest


@pytest.fixture(params=[3, 4] if sys.version_info < (3, 8) else [3, 4, 5])
def pickle_protocol(request) -> int:
    return request.param
