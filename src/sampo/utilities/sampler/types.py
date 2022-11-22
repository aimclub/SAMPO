from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')


@dataclass(frozen=True)
class MinMax(Generic[T]):
    min: T
    max: T

    def __post_init__(self):
        assert type(self.min) != T
        assert type(self.max) != T
