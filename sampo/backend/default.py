from typing import Callable, Any, Type

from sampo.backend import ComputationalBackend, T, R, ComputationalContext

class DefaultComputationalContext(ComputationalContext):

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return [action(v) for v in values]


class DefaultComputationalBackend(ComputationalBackend):

    def register(self, key: Any, action: Callable[[ComputationalContext, tuple], Any]):
        pass

    def run(self, key: Any, args: tuple, return_type: Type[T]) -> T:
        pass

    def new_context(self) -> ComputationalContext:
        return DefaultComputationalContext()
