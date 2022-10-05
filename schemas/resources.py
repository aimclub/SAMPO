from dataclasses import dataclass
from random import Random
from typing import Optional

from schemas.identifiable import Identifiable
from schemas.interval import IntervalGaussian
from schemas.serializable import AutoJSONSerializable
from schemas.types import AgentId


@dataclass
class Resource(AutoJSONSerializable['Equipment'], Identifiable):
    pass


@dataclass
class Worker(Resource):
    count: int
    contractor_id: Optional[str] = ""
    productivity_class: Optional[int] = 0
    productivity: Optional[IntervalGaussian] = IntervalGaussian(1, 0, 1, 1)

    ignored_fields = ['productivity']

    def copy(self):
        return Worker(id=self.id,
                      name=self.name,
                      count=self.count,
                      contractor_id=self.contractor_id,
                      productivity_class=self.productivity_class,
                      productivity=self.productivity)

    def get_agent_id(self) -> AgentId:
        return self.contractor_id, self.name

    def get_static_productivity(self) -> float:
        return self.productivity.mean * self.count

    def get_stochastic_productivity(self, rand: Optional[Random] = None) -> float:
        return self.productivity.float(rand) * self.count

    def __repr__(self):
        return str(self.count)


@dataclass
class ConstructionObject(Resource):
    pass


@dataclass(init=False)
class EmptySpaceConstructionObject(ConstructionObject):
    id: str = "00000000000000000"
    name: str = "empty space construction object"


@dataclass
class Equipment(Resource):
    pass


@dataclass
class Material(Resource):
    pass
