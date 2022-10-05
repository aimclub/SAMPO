from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from schemas.serializable import AutoJSONSerializable


class BaseReq(AutoJSONSerializable['BaseReq'], ABC):
    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        ...


WORKER_INF = 100  # int(1e11)


@dataclass(frozen=True)
class WorkerReq(BaseReq):
    type: str
    volume: float
    min_count: Optional[int] = 1
    max_count: Optional[int] = WORKER_INF
    name: Optional[str] = None

    def scale(self, scalar: float, new_name: Optional[str] = None) -> 'WorkerReq':
        """
        The function scales the requirement to the size of the work including the total
        volume and the maximum number of personnel involved.
        :param scalar: float - scalar for multiplication
        :param new_name: str - name for new req
        :return new_req: WorkerReq - new object with new volume of the work and extended max_count_commands
        """
        max_count = self.max_count
        if max_count < WORKER_INF:
            max_count = max(round(self.max_count * scalar), self.min_count)

        return WorkerReq(self.type, self.volume * scalar,
                         self.min_count, max_count, new_name)

    def mul_volume(self, scalar, new_name: Optional[str] = None) -> 'WorkerReq':
        """
        The function scales only volume of the work for the requirement
        :param scalar: float - scalar for multiplication
        :param new_name: str - name for new req
        :return: WorkerReq - new object with new volume of the work.
        """
        new_name = new_name or self.name
        return WorkerReq(self.type, self.volume * scalar,
                         self.min_count, self.max_count, new_name)


@dataclass(frozen=True)
class EquipmentReq(BaseReq):
    type: str
    name: Optional[str] = None


@dataclass(frozen=True)
class MaterialReq(BaseReq):
    type: str
    name: Optional[str] = None


@dataclass(frozen=True)
class ConstructionObjectReq(BaseReq):
    type: str
    name: Optional[str] = None
