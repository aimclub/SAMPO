"""Common type utilities for sampler.

Common type utilities for sampler.
Общие типовые утилиты для выборки.
"""

from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass(frozen=True)
class MinMax(Generic[T]):
    """Range with minimum and maximum values.

    Range with minimum and maximum values.
    Диапазон с минимальным и максимальным значениями.

    Attributes:
        min: Lower bound. min: Нижняя граница.
        max: Upper bound. max: Верхняя граница.
    """

    min: T
    max: T
