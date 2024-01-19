from abc import ABC, abstractmethod
from typing import Callable, TypeVar

T = TypeVar('T')
R = TypeVar('R')

class ComputationalContext(ABC):
    @abstractmethod
    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        ...



class ComputationalBackend(ABC):

    def __init__(self):
        self._context = self.new_context()

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._context.map(action, values)

    def set_context(self, context: ComputationalContext):
        self._context = context

    @abstractmethod
    def new_context(self) -> ComputationalContext:
        ...
