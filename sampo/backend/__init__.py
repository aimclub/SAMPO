from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Any, Type

T = TypeVar('T')
R = TypeVar('R')

class ComputationalContext(ABC):
    @abstractmethod
    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        ...



class ComputationalBackend(ABC):

    def __init__(self):
        self._context = self.new_context()
        self._actions = {}

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._context.map(action, values)

    def set_context(self, context: ComputationalContext):
        self._context = context

    @abstractmethod
    def new_context(self) -> ComputationalContext:
        ...

    def register(self, key: Any, action: Callable[[ComputationalContext, tuple], Any]):
        self._actions[key] = action

    # `return_type` is here for type inference
    def run(self, key: Any, args: tuple, return_type: Type[T]) -> T:
        return self._actions[key](self._context, *args)
