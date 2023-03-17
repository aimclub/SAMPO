from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Resource


class ResourceHolder(Resource):
    def __init__(self, id: str, name: str, productivity: IntervalGaussian):
        super(ResourceHolder, self).__init__(id, name)
        self._productivity = productivity

    @property
    def productivity(self):
        return self._productivity

    def copy(self):
        return ResourceHolder(self.id, self.name, self.productivity)

class Road(Resource):
    def __init__(self, id: str, name: str, throughput: IntervalGaussian):
        super(Road, self).__init__(id, name)
        self._throughput = throughput

    @property
    def throughput(self):
        return self._throughput

    def copy(self):
        return Road(self.id, self.name, self.throughput)

class LandscapeConfiguration:
    def __init__(self, roads: list[Road], holders: list[ResourceHolder]):
        self._roads = roads
        self._holders = holders

    def get_all_resources(self) -> list[Resource]:
        return self._roads + self._holders
