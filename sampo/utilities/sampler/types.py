from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')


@dataclass(frozen=True)
class MinMax(Generic[T]):
    min: T
    max: T
