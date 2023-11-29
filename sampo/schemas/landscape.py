import heapq
from abc import ABC, abstractmethod
from collections import defaultdict

from sampo.schemas.landscape_graph import LandGraph, LandGraphNode
from sampo.schemas.resources import Resource, Material
from sampo.schemas.sorted_list import ExtendedSortedList
from sampo.schemas.time import Time
from sampo.schemas.zones import ZoneConfiguration

WAY_LENGTH = 10000000

class ResourceSupply(Resource, ABC):
    def __init__(self, id: str, name: str, count: int):
        super(ResourceSupply, self).__init__(id, name, count)

    @abstractmethod
    def get_available_resources(self) -> list[tuple[int, str]]:
        ...


class Vehicle(ResourceSupply):
    def __init__(self,
                 id: str,
                 name: str,
                 holder: 'ResourceHolder',
                 capacity: list[Material]):
        super(Vehicle, self).__init__(id, name, 0)
        self.holder = holder
        self.capacity = capacity
        self.cost = 0.0

    def get_available_resources(self) -> list[tuple[int, str]]:
        return [(mat.count, mat.name) for mat in self.capacity]


class ResourceHolder(ResourceSupply):
    def __init__(self,
                 name: str,
                 vehicles: list[Vehicle] = None,
                 node: LandGraphNode = None):
        """
        :param name:
        :param vehicles:
        :param node:
        """
        # let ids of two objects will be similar to make simpler matching ResourceHolder to node in LandGraph
        super(ResourceHolder, self).__init__(node.id, name, 0)
        self.vehicles = vehicles
        self.node = node

    def get_available_resources(self) -> list[tuple[int, str]]:
        return [(mat.count, mat.name) for mat in self.node.resource_storage_unit.get_capacity()]


class LandscapeConfiguration:
    def __init__(self,
                 holders: list[ResourceHolder] = None,
                 lg: LandGraph = None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        self.routing_mx = None
        if holders is None:
            holders = []
        self.lg = lg
        self._holders: list[ResourceHolder] = holders

        # _ind2holder_id is required to match ResourceHolder's id to index in list of LangGraphNodes to work with routing_mx
        self._ind2holder_id = {self.lg.node2ind[holder.node]: holder.node.id for holder in self._holders}
        self.holder_id2resource_holder = {holder.node.id: holder for i, holder in enumerate(self._holders)}
        self.zone_config = zone_config

    def build_landscape(self):
        self.routing_mx = [[[WAY_LENGTH, defaultdict(set)] for j in range(self.lg.vertex_count)] for i in
                           range(self.lg.vertex_count)]
        self._build_routes()

    def get_sorted_holders(self, node_id: int) -> ExtendedSortedList[tuple[list[float, dict[set]], str]]:
        """
        :param node_id: id of node in LandGraph's list of nodes
        :return: sorted list of holders' id by the length of way
        """
        holders = []
        for i in range(len(self.routing_mx)):
            if int(self.routing_mx[node_id][i][0]) != WAY_LENGTH:
                holders.append((self.routing_mx[node_id][i], self._ind2holder_id[i]))
        return ExtendedSortedList(holders, key=lambda x: x[0][0])

    def get_holders(self) -> list[ResourceSupply]:
        return self._holders

    def get_roads(self):
        return self.lg.roads

    def _build_routes(self):
        def dijkstra(holder_id: int):
            visited = [False] * len(self.lg.nodes)
            visited[holder_id] = True
            priority_queue = [(0.0, holder_id)]

            while len(priority_queue) > 0:
                current_dist, current_v = heapq.heappop(priority_queue)

                if current_dist > self.routing_mx[current_v][holder_id][0]:
                    continue

                for road in self.lg.nodes[current_v].roads:
                    dist: float = current_dist + road.weight
                    if dist < self.routing_mx[self.lg.node2ind[road.finish]][holder_id][0]:
                        self.routing_mx[self.lg.node2ind[road.finish]][holder_id][0] = dist
                        self.routing_mx[self.lg.node2ind[road.finish]][holder_id][1][holder_id] = self.routing_mx[current_v][holder_id][1][holder_id].copy()
                        self.routing_mx[self.lg.node2ind[road.finish]][holder_id][1][holder_id].add(current_v)
                        heapq.heappush(priority_queue, (dist, self.lg.node2ind[road.finish]))

        holder_indices = [self.lg.node2ind[holder.node] for holder in self._holders]
        for i in holder_indices:
            dijkstra(i)


class MaterialDelivery:
    def __init__(self, work_id: str):
        self.id = work_id
        self.delivery = {}

    def add_deliveries(self, name: str, deliveries: list[tuple[Time, int]]):
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            self.delivery[name] = deliveries
        else:
            material_delivery.extend(deliveries)
