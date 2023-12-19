import math
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from sortedcontainers import SortedList

from sampo.schemas.landscape_graph import LandGraph, LandGraphNode, LandEdge
from sampo.schemas.resources import Material
from sampo.schemas.zones import ZoneConfiguration


class ResourceSupply(ABC):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    @abstractmethod
    def get_resources(self) -> list[tuple[str, int]]:
        ...


class Road(ResourceSupply):
    def __init__(self, name: str,
                 edge: LandEdge,
                 speed: float = 50):
        """
        :param name: name of road
        :param edge: the edge in LandGraph
        :poram bandwidth: the number of vehicles that road can pass per hour
        :param speed: the maximum value of speed on the road
        :param vehicles: the number of vehicles that are on the road at the current moment
        """
        super(Road, self).__init__(edge.id, name)
        self.vehicles = edge.bandwidth
        self.length = edge.weight
        self.speed = speed
        self.overcome_time = math.ceil(self.length / self.speed)
        self.edge = edge

    def get_resources(self) -> list[tuple[str, int]]:
        return [('speed', self.speed), ('length', self.edge.weight), ('vehicles', self.vehicles)]


class Vehicle(ResourceSupply):
    def __init__(self,
                 id: str,
                 name: str,
                 capacity: list[Material]):
        super(Vehicle, self).__init__(id, name)
        self.capacity = capacity
        self.cost = 0.0
        self.volume = sum([mat.count for mat in self.capacity])

    @cached_property
    def resources(self) -> dict[str, int]:
        return {mat.name: mat.count for mat in self.capacity}

    def get_resources(self) -> list[tuple[str, int]]:
        return [(mat.name, mat.count) for mat in self.capacity]

    def get_sum_resources(self) -> int:
        return sum([mat.count for mat in self.capacity])


class ResourceHolder(ResourceSupply):
    def __init__(self,
                 id: str,
                 name: str,
                 vehicles: list[Vehicle] = None,
                 node: LandGraphNode = None):
        """
        :param name:
        :param vehicles:
        :param node:
        """
        # let ids of two objects will be similar to make simpler matching ResourceHolder to node in LandGraph
        super(ResourceHolder, self).__init__(id, name)
        self.node_id = node.id
        self.vehicles = vehicles
        self.node = node

    def get_vehicles_resources(self) -> list[list[tuple[str, int]]]:
        return [vehicle.get_resources() for vehicle in self.vehicles]

    def get_resources(self) -> list[tuple[str, int]]:
        return [(name, count) for name, count in self.node.resource_storage_unit.capacity.items()]


class LandscapeConfiguration:
    def __init__(self,
                 holders: list[ResourceHolder] = None,
                 lg: LandGraph = None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        self.WAY_LENGTH = np.Inf
        self.dist_mx: list[list[float]] = None
        self.path_mx: np.array = None
        self.road_mx: list[list[str]] = None
        if holders is None:
            holders = []
        self.lg: LandGraph = lg
        self._holders: list[ResourceHolder] = holders

        # _ind2holder_id is required to match ResourceHolder's id to index in list of LangGraphNodes to work with routing_mx
        self.ind2holder_id: dict[int, str] = {self.lg.node2ind[holder.node]: holder.node.id for holder in
                                              self._holders}
        self.holder_id2resource_holder: dict[str, ResourceHolder] = {holder.node.id: holder for holder in self._holders}
        self.zone_config = zone_config

    def build_landscape(self):
        self._build_routes()

    def get_sorted_holders(self, node_id: int) -> SortedList[list[tuple[float, str]]]:
        """
        :param node_id: id of node in LandGraph's list of nodes
        :return: sorted list of holders' id by the length of way
        """
        holders = []
        for i in self.ind2holder_id.keys():
            if self.dist_mx[node_id][i] != self.WAY_LENGTH:
                holders.append((self.dist_mx[node_id][i], self.ind2holder_id[i]))
        return SortedList(holders, key=lambda x: x[0])

    @cached_property
    def holders(self) -> list[ResourceHolder]:
        return self._holders

    @cached_property
    def platforms(self) -> list[LandGraphNode]:
        platform_ids = set([node.id for node in self.lg.nodes]).difference(
            set([holder.node_id for holder in self._holders]))
        return [node for node in self.lg.nodes if node.id in platform_ids]

    @cached_property
    def roads(self) -> list[Road]:
        return [Road(f'road_{i}', edge) for i, edge in enumerate(self.lg.edges)]

    def get_all_resources(self) -> list[dict]:
        def merge_dicts(a, b):
            c = a.copy()
            c.update(b)
            return c

        holders = {
            holder.id: merge_dicts({name: count for name, count in holder.node.resource_storage_unit.capacity.items()},
                                   {'vehicles': len(holder.vehicles)})
            for holder in self.holders}
        roads = {road.id: {'vehicles': road.vehicles}
                 for road in self.roads}
        platforms = {node.id: {name: count for name, count in node.resource_storage_unit.capacity.items()}
                     for node in self.platforms}

        resources = [holders, roads, platforms]
        return resources

    def _build_routes(self):
        count = self.lg.vertex_count
        dist_mx = self.lg.adj_matrix.copy()
        path_mx: np.array = np.full((count, count), -1)
        road_mx: list[list[str]] = [['-1' for j in range(count)] for i in range(count)]

        for v in range(count):
            for u in range(count):
                if v == u:
                    path_mx[v][u] = 0
                elif dist_mx[v][u] != np.Inf:
                    path_mx[v][u] = v
                    for road in self.lg.nodes[v].roads:
                        if self.lg.node2ind[road.finish] == u:
                            road_mx[v][u] = road.id
                else:
                    path_mx[v][u] = -1

        for i in range(self.lg.vertex_count):
            for u in range(self.lg.vertex_count):
                for v in range(self.lg.vertex_count):
                    if (dist_mx[u][i] != np.Inf and dist_mx[u][i] != 0
                            and dist_mx[i][v] != np.Inf and dist_mx[i][v] != 0
                            and dist_mx[u][i] + dist_mx[i][v] < dist_mx[u][v]):
                        dist_mx[u][v] = dist_mx[u][i] + dist_mx[i][v]
                        path_mx[u][v] = path_mx[i][v]

        self.dist_mx = dist_mx
        self.path_mx = path_mx
        self.road_mx = road_mx
