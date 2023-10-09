from dataclasses import dataclass
from typing import TypeVar, Iterable, Generic

T = TypeVar('T')


@dataclass
class Node(Generic[T]):
    value: T
    next: 'Node' = None


class Iterator(Generic[T]):
    def __init__(self, lst: 'LinkedList'):
        self._lst = lst
        self._node = lst._root
        self._prev = None

    def __iter__(self):
        yield self._node

        while self._node is not None:
            self._prev = self._node
            self._node = self._node.next
            yield self._node

    def remove(self) -> T:
        node = self._node
        next = node.next
        self._prev.next = next
        self._node = next
        return node.value


class LinkedList(Generic[T]):
    def __init__(self, iterable: Iterable[T] | None = None):
        self._root = None
        self._tail = None
        if iterable:
            for v in iterable:
                self.append(v)

    def append(self, v: T):
        if self._root is None:
            self._root = Node(v)
            self._tail = self._root
        else:
            old_tail = self._tail
            self._tail = Node(v)
            old_tail.next = self._tail

    def __add__(self, other: T):
        self.append(other)

    def iterator(self):
        return Iterator(self)

    def __iter__(self):
        return (node.value for node in self.iterator())
