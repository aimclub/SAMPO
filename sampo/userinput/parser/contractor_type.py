"""Definitions of contractor capability tiers.

Определения уровней возможностей подрядчиков."""

from enum import Enum

# pylint: disable=invalid-name


class ContractorType(Enum):
    """Levels of contractor performance.

    Уровни производительности подрядчика.
    """

    Minimal = 50
    Average = 75
    Maximal = 200

    def command_capacity(self) -> int:
        """Return command capacity in man-hours.

        Возвращает командную мощность в человеко-часах.
        """

        return self.value
