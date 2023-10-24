from abc import ABC, abstractmethod
from copy import deepcopy

from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Resource, Material
from sampo.schemas.time import Time
from sampo.schemas.zones import ZoneConfiguration


class ResourceSupply(Resource, ABC):
    def __init__(self, id: str, name: str, count: int):
        super(ResourceSupply, self).__init__(id, name, count)

    @abstractmethod
    def get_available_resources(self) -> list[tuple[int, str]]:
        ...


class ResourceHolder(ResourceSupply):
    def __init__(self, id: str, name: str, productivity: IntervalGaussian, materials: list[Material]):
        super(ResourceHolder, self).__init__(id, name, int(productivity.mean))
        self._productivity = productivity
        self._materials = materials

    @property
    def productivity(self):
        return self._productivity

    def copy(self):
        return ResourceHolder(self.id, self.name, self.productivity, deepcopy(self._materials))

    def get_available_resources(self) -> list[tuple[int, str]]:
        return [(mat.count, mat.name) for mat in self._materials]


class Road(ResourceSupply):
    def __init__(self, id: str, name: str, throughput: IntervalGaussian):
        super(Road, self).__init__(id, name, int(throughput.mean))
        self._throughput = throughput

    @property
    def throughput(self):
        return self._throughput

    def copy(self):
        return Road(self.id, self.name, self.throughput)

    def get_available_resources(self) -> list[tuple[int, str]]:
        return []


class LandscapeConfiguration:
    def __init__(self, roads=None,
                 holders=None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        if holders is None:
            holders = []
        if roads is None:
            roads = []
        self._roads = roads
        self._holders = holders
        self.zone_config = zone_config

    def get_all_resources(self) -> list[ResourceSupply]:
        return self._roads + self._holders


class MaterialDelivery:
    def __init__(self, work_id: str):
        self.id = work_id
        self.delivery = {}

    def add_delivery(self, name: str, time: Time, count: int):
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            material_delivery = []
            self.delivery[name] = material_delivery
        material_delivery.append((time, count))

    def add_deliveries(self, name: str, deliveries: list[tuple[Time, int]]):
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            self.delivery[name] = deliveries
        else:
            material_delivery.extend(deliveries)
