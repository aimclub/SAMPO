from copy import deepcopy
from typing import Tuple

from schemas.sorted_list import ExtendedSortedList


def merger(old: Tuple[int, int], new: Tuple[int, int]) -> Tuple[int, int]:
    return old[0], old[1] + new[1]


def test_merge():
    """
    Here is the test for efficient version of `merge` operation.
    This test was made to measure performance and compare to old variant.
    """
    for size in range(9990, 10000):
        for inc in range(10):
            old_lst = ExtendedSortedList(iterable=[(i, 1) for i in range(size)], key=lambda x: x[0])
            lst = deepcopy(old_lst)
            for i in range(size):
                lst.merge(lst[i], (i, inc), merger)

            for i in range(size):
                assert lst[i] == (old_lst[i][0], old_lst[i][1] + inc)


def _test_merge_old():
    """
    Here is the test for manual version of `merge` operation, e.g. remove-and-add.
    This test was made to measure performance and compare to new variant.
    """
    for size in range(9900, 10000):
        for inc in range(10):
            old_lst = ExtendedSortedList(iterable=[(i, 1) for i in range(size)], key=lambda x: x[0])
            lst = deepcopy(old_lst)
            for i in range(size):
                # lst.merge(lst[i], (i, inc), merger)
                old = lst[i]
                lst.remove(old)
                lst.add((i, old[1] + inc))

            for i in range(size):
                assert lst[i] == (old_lst[i][0], old_lst[i][1] + inc)


