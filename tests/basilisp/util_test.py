from basilisp.util import partition


def test_partition():
    assert [(1, 2, 3, 4)] == list(partition([1, 2, 3, 4], 5))
    assert [(1, 2, 3, 4)] == list(partition([1, 2, 3, 4], 4))
    assert [(1, 2, 3), (4,)] == list(partition([1, 2, 3, 4], 3))
    assert [(1, 2), (3, 4)] == list(partition([1, 2, 3, 4], 2))
    assert [(1,), (2,), (3,), (4,)] == list(partition([1, 2, 3, 4], 1))
