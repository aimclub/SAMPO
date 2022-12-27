import uuid
from random import Random
from typing import Optional
from uuid import uuid4



# TODO: describe the function (description, parameters, return type)
def uuid_str(rand: Optional[Random] = None) -> str:
    ans = uuid4() if rand is None else uuid.UUID(int=rand.getrandbits(128))
    return str(ans)


# TODO: describe the function (description, parameters, return type)


"""
# TODO: describe the function (description, parameters, return type)
def binary_search(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# TODO: describe the function (description, parameters, return type)
def binary_search_reversed(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] > x:
            lo = mid + 1
        else:
            hi = mid
    return lo
"""
