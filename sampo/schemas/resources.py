from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Optional

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.types import AgentId


class WorkerProductivityMode(Enum):
    Static = 'static',
    Stochastic = 'stochastic'


@dataclass
class Resource(AutoJSONSerializable['Resource'], Identifiable):
    """
    A class summarizing the different resources used in the work: Human resources, equipment, materials, etc.
    """
    id: str
    name: str
    count: int
    contractor_id: Optional[str] = ""

    def __init__(self,
                 id: str,
                 name: str,
                 count: int,
                 contractor_id: Optional[str] = ""):
        self.id = id
        self.name = name
        self._count = count
        self.contractor_id = contractor_id

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = int(value)

    # TODO: describe the function (description, return type)
    def get_agent_id(self) -> AgentId:
        return self.contractor_id, self.name


@dataclass
class Worker(Resource):
    """
    A class dedicated to human resources

    :param count: the number of people in this resource
    :param contractor_id: Contractor id if resources are added directly to the contractor
    :param productivity: interval from Gaussian or Uniform distribution, that contains possible values of
    productivity of certain worker
    """

    def __init__(self,
                 id: str,
                 name: str,
                 count: int,
                 contractor_id: Optional[str] = "",
                 productivity: Optional[IntervalGaussian] = IntervalGaussian(1, 0, 1, 1),
                 cost_one_unit: Optional[float] = None):
        super(Worker, self).__init__(id, name, int(count), contractor_id)
        self.productivity = productivity if productivity is not None else IntervalGaussian(1, 0, 1, 1)
        self.cost_one_unit = cost_one_unit if cost_one_unit is not None else self.productivity.mean * 10

    ignored_fields = ['productivity']

    def copy(self):
        """
        Return copied current object

        :return: object of Worker class
        """
        return Worker(id=self.id,
                      name=self.name,
                      count=self.count,
                      contractor_id=self.contractor_id,
                      productivity=self.productivity)

    def with_count(self, count: int) -> 'Worker':
        """
        Update count field

        :param count: amount of current type of worker
        :return: upgraded current object
        """
        self.count = int(count)
        return self

    def get_cost(self) -> float:
        """Returns cost of this worker entry"""
        return self.cost_one_unit * int(self.count)

    def get_agent_id(self) -> AgentId:
        """
        Return the agent unique identifier in schedule

        :return: contractor's id and name
        """
        return self.contractor_id, self.name

    def get_productivity(self, rand: Random, productivity_mode: WorkerProductivityMode) -> float:
        """
        Return the productivity of the worker team.
        It has 2 mods: stochastic and non-stochastic.

        :param productivity_mode:
        :param rand: parameter for stochastic part
        """
        if productivity_mode is WorkerProductivityMode.Static:
            return self.productivity.mean * self.count
        return self.productivity.rand_float(rand) * self.count

    def __repr__(self):
        return f'{self.count} {self.name}'

    def __str__(self):
        return self.__repr__()


@dataclass
class ConstructionObject(Resource):
    pass


@dataclass(init=False)
class EmptySpaceConstructionObject(ConstructionObject):
    id: str = '00000000000000000'
    name: str = 'empty space construction object'


@dataclass
class Equipment(Resource):
    pass


@dataclass
class Material(Resource):

    def __init__(self,
                 id: str,
                 name: str,
                 count: int,
                 cost_one_unit: Optional[float] = 1):
        super(Material, self).__init__(id, name, count)
        self.cost_one_unit = cost_one_unit

    # TODO: describe the function (description, return type)
    def copy(self):
        return Material(id=self.id,
                        name=self.name,
                        count=self.count)

    def with_count(self, count: int) -> 'Material':
        self.count = count
        return self
