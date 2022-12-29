from dataclasses import dataclass
from random import Random
from typing import Optional

import numpy as np

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.types import AgentId


@dataclass
class Resource(AutoJSONSerializable['Equipment'], Identifiable):
    """
    A class summarizing the different resources used in the work: Human resources, equipment, materials, etc.
    """
    pass


# TODO describe "productivity"
@dataclass
class Worker(Resource):
    """
    A class dedicated to human resources
    :param count: the number of people in this resource
    :param contractor_id: Contractor id if resources are added directly to the contractor
    :param productivity: Contractor id if resources are added directly to the contractor
    """

    def __init__(self,
                 id: str,
                 name: str,
                 count: int,
                 contractor_id: Optional[str] = "",
                 productivity: Optional[IntervalGaussian] = IntervalGaussian(1, 0, 1, 1),
                 cost_one_unit: Optional[float] = None):
        super(Worker, self).__init__(id, name)
        self.count = count
        self.contractor_id = contractor_id
        self.productivity = productivity if productivity is not None else IntervalGaussian(1, 0, 1, 1)
        self.cost_one_unit = cost_one_unit if cost_one_unit is not None else self.productivity.mean * 10

    ignored_fields = ['productivity']

    # TODO: describe the function (description, return type)
    def copy(self):
        return Worker(id=self.id,
                      name=self.name,
                      count=self.count,
                      contractor_id=self.contractor_id,
                      productivity=self.productivity)

    def with_count(self, count: int) -> 'Worker':
        self.count = count
        return self

    def get_cost(self) -> float:
        """Returns cost of this worker entry"""
        return self.cost_one_unit * self.count

    # TODO: describe the function (description, return type)
    def get_agent_id(self) -> AgentId:
        return self.contractor_id, self.name

    # TODO: describe the function (description, return type)
    def get_static_productivity(self) -> float:
        return self.productivity.mean * self.count

    # TODO: describe the function (description, return type)
    def get_stochastic_productivity(self, rand: Optional[Random] = None) -> float:
        return self.productivity.rand_float(rand) * self.count

    # TODO: describe the function (description, return type)
    def __repr__(self):
        return f'{self.count} {self.name}'

# TODO: describe the class (description)
@dataclass
class ConstructionObject(Resource):
    pass

# TODO: describe the class (description, parameters)
@dataclass(init=False)
class EmptySpaceConstructionObject(ConstructionObject):
    id: str = "00000000000000000"
    name: str = "empty space construction object"

# TODO: describe the class (description)
@dataclass
class Equipment(Resource):
    pass

# TODO: describe the class (description)
@dataclass
class Material(Resource):
    pass
