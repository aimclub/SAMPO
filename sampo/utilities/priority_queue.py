from typing import TypeVar, List, Callable, Generic

from sortedcontainers import SortedKeyList

T = TypeVar('T')


class PriorityQueue(Generic[T]):
    _h: SortedKeyList
    _key_getter: Callable[[T], float]

    def __init__(self, lst: List[T], descending: bool = False, key_getter: Callable[[T], float] = lambda x: x):
        comparator = (lambda x: -key_getter(x[1])) if descending else (lambda x: key_getter(x[1]))
        self._h = SortedKeyList([(key_getter(v), v) for v in lst], comparator)
        self._key_getter = key_getter

    @staticmethod
    def empty(descending: bool = False, key_getter: Callable[[T], float] = lambda x: x):
        return PriorityQueue([], descending, key_getter)

    def add(self, value: T):
        self._h.add((self._key_getter(value), value))

    def extract_extremum(self) -> T:
        return self._h.pop()[1]

    def replace(self, old: T, new: T):
        self._h.discard((self._key_getter(old), old))
        self._h.add((self._key_getter(new), new))

    def decrease_key(self, value: T):
        self.replace(value, value)

    def __len__(self):
        return len(self._h)
