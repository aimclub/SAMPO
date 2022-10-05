from collections.abc import Iterable
from typing import List, Any, Union


# Returns a generator which should flatten any heterogeneous iterable
def flatten(xs: Iterable[Union[List[Any], Any]]) -> Iterable[Any]:
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
