from dataclasses import dataclass
from typing import TypeVar, Iterable, Generic, Callable

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
            yield self.__next__()

    def __next__(self) -> Node[T]:
        return self.next()

    def next(self) -> Node[T]:
        self._prev = self._node
        self._node = self._node.next
        return self._node

    def get(self) -> Node[T]:
        return self._node

    def has_next(self) -> bool:
        return self._node is not None

    def remove(self) -> T:
        if self._node == self._lst._root:
            old = self._lst._root
            self._lst._root = old.next
            self._node = self._lst._root
            self._lst._len -= 1
            return old.value
        node = self._node
        next = node.next
        self._prev.next = next
        self._node = next
        self._lst._len -= 1
        return node.value


class LinkedList(Generic[T]):
    def __init__(self, iterable: Iterable[T] | None = None):
        self._root = None
        self._tail = None
        self._len = 0
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
        self._len += 1

    def remove_if(self, condition: Callable[[T], bool]):
        it = self.iterator()

        while it.has_next():
            v = it.get()
            if condition(v.value):
                it.remove()
            else:
                it.next()

    def __add__(self, other: T):
        self.append(other)

    def iterator(self):
        return Iterator(self)

    def __iter__(self):
        return (node.value for node in self.iterator())

    def __len__(self):
        return self._len

    def is_empty(self):
        return len(self) == 0
