from typing import Callable

from sampo.backend import ComputationalBackend, T, R, ComputationalContext


class DefaultComputationalContext(ComputationalContext):

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return [action(v) for v in values]


class DefaultComputationalBackend(ComputationalBackend):
    _actions = {}

    def new_context(self) -> ComputationalContext:
        return DefaultComputationalContext()
