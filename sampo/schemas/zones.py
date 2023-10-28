from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from sampo.schemas.time import Time
from sampo.schemas.serializable import AutoJSONSerializable


@dataclass
class Zone:
    name: str
    status: int


class ZoneStatuses(ABC):
    @abstractmethod
    def statuses_available(self) -> int:
        """
        :return: number of statuses available
        """
        ...

    @abstractmethod
    def match_status(self, target: int, to_compare: int) -> bool:
        """
        :param target: statuses that should match
        :param to_compare: status that should be matched
        :return: does target match to_compare
        """
        ...


class DefaultZoneStatuses(ZoneStatuses):
    """
    Statuses: 0 - not stated, 1 - opened, 2 - closed
    """

    def statuses_available(self) -> int:
        return 3

    def match_status(self, status_to_check: int, required_status: int) -> bool:
        return required_status == 0 or status_to_check == required_status


@dataclass
class ZoneConfiguration:
    start_statuses: dict[str, int] = field(default_factory=dict)
    time_costs: np.ndarray = field(default_factory=lambda: np.array([[]]))
    statuses: ZoneStatuses = field(default_factory=lambda: DefaultZoneStatuses())

    def change_cost(self, from_status: int, to_status: int):
        return self.time_costs[from_status, to_status]


@dataclass
class ZoneTransition(AutoJSONSerializable['ZoneTransition']):
    name: str
    from_status: int
    to_status: int
    start_time: Time
    end_time: Time
