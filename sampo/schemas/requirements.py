from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time

# Used for max_count in the demand, if it is not specified during initialization WorkerReq
DEFAULT_MAX_COUNT = 100


class BaseReq(AutoJSONSerializable['BaseReq'], ABC):
    """
    A class summarizing any requirements for the work to be performed related to renewable and non-renewable
    resources, infrastructure requirements, etc.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the claim if it exists, e.g. 'dig work claim'.
        :return name: the name of req
        """
        ...


@dataclass(frozen=True)
class WorkerReq(BaseReq):
    """
    Requirements related to renewable human resources
    :param kind: type of resource/profession
    :param volume: volume of work in time units
    :param min_count: minimum number of employees needed to perform the work
    :param max_count: maximum allowable number of employees performing work
    :param name: the name of this requirement
    """
    kind: str
    volume: Time
    min_count: Optional[int] = 1
    max_count: Optional[int] = DEFAULT_MAX_COUNT
    name: Optional[str] = ''

    def scale_all(self, scalar: float, new_name: Optional[str] = '') -> 'WorkerReq':
        """
        The function scales the requirement to the size of the work including the total
        volume and the maximum number of personnel involved.
        :param scalar: scalar for multiplication
        :param new_name: name for new req
        :return new_req: new object with new volume of the work and extended max_count_commands
        """
        max_count = max(round(self.max_count * scalar), self.min_count)
        new_req = WorkerReq(self.kind, self.volume * scalar, self.min_count, max_count, new_name or self.name)
        return new_req

    def scale_volume(self, scalar: float, new_name: Optional[str] = None) -> 'WorkerReq':
        """
        The function scales only volume of the work for the requirement.
        :param scalar: scalar for multiplication
        :param new_name: name for new req
        :return new_req: new object with new volume of the work.
        """
        new_req = WorkerReq(self.kind, self.volume * scalar, self.min_count, self.max_count, new_name or self.name)
        return new_req


@dataclass(frozen=True)
class EquipmentReq(BaseReq):
    """
    Requirements for renewable non-human resources: equipment, trucks, machines, etc
    :param kind: type of resource/profession
    :param name: the name of this requirement
    """
    kind: str
    name: Optional[str] = None


@dataclass(frozen=True)
class MaterialReq(BaseReq):
    """
    Requirements for non-renewable materials: consumables, spare parts, construction materials
    :param kind: type of resource/profession
    :param name: the name of this requirement
    """
    kind: str
    name: Optional[str] = None


@dataclass(frozen=True)
class ConstructionObjectReq(BaseReq):
    """
    Requirements for infrastructure and the construction of other facilities: electricity, pipelines, roads, etc
    :param kind: type of resource/profession
    :param name: the name of this requirement
    """
    kind: str
    name: Optional[str] = None
