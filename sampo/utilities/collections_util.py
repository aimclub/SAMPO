from collections.abc import Iterable
from typing import Union, TypeVar, Callable

T = TypeVar('T')


def flatten(xs: Iterable[Union[list[T], T]]) -> Iterable[T]:
    """
    Returns a generator which should flatten any heterogeneous iterable

    :param xs:
    :return:
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


K = TypeVar('K')
V = TypeVar('V')


def build_index(items: Iterable[T], key_getter: Callable[[T], K], value_getter: Callable[[T], V] = lambda x: x) \
        -> dict[K, V]:
    """
    :param items: an iterable to index
    :param key_getter: a function that should retrieve index key from item
    :param value_getter: a function that should retrieve index value from item
    :return: dictionary that represents built index given by `key_getter` function
    """
    return {key_getter(item): value_getter(item) for item in items}


def reverse_dictionary(dictionary: dict[K, V]) -> dict[V, K]:
    return {value: key for key, value in dictionary.items()}
