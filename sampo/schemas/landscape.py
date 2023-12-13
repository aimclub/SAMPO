import heapq
from abc import ABC, abstractmethod
from functools import cached_property

from sortedcontainers import SortedList

from sampo.schemas.landscape_graph import LandGraph, LandGraphNode, LandEdge
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.zones import ZoneConfiguration


class ResourceSupply(ABC):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    @abstractmethod
    def get_resources(self) -> list[tuple[int, str]]:
        ...


class Road(ResourceSupply):
    def __init__(self, name: str,
                 edge: LandEdge,
                 speed: float = 50,
                 vehicles: int = 0):
        """
        :param name: name of road
        :param count: road capacity
        :param edge: the edge in LandGraph
        :param speed: the maximum value of speed on the road
        :param vehicles: the number of vehicles that are on the road at the current moment
        """
        super(Road).__init__(edge.id, name)
        self.vehicles = vehicles
        self.speed = speed
        self.edge = edge

    def get_resources(self) -> list[tuple[int, str]]:
        return [(self.speed, 'speed'), (self.edge.weight, 'length'), (self.vehicles, 'vehicles')]


class Vehicle(ResourceSupply):
    def __init__(self,
                 id: str,
                 name: str,
                 holder: 'ResourceHolder',
                 capacity: list[Material]):
        super(Vehicle, self).__init__(id, name)
        self.holder = holder
        self.capacity = capacity
        self.cost = 0.0
        self.volume = sum([mat.count for mat in self.capacity])

    @cached_property
    def resources(self) -> dict[str, int]:
        return {mat.name: mat.count for mat in self.capacity}

    def get_resources(self) -> list[tuple[int, str]]:
        return [(mat.count, mat.name) for mat in self.capacity]

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

    def get_vehicles_resources(self) -> list[list[tuple[int, str]]]:
        return [vehicle.get_resources() for vehicle in self.vehicles]

    def get_resources(self) -> list[tuple[int, str]]:
        return [(count, name) for name, count in self.node.resource_storage_unit.capacity.items()]


class LandscapeConfiguration:
    def __init__(self,
                 holders: list[ResourceHolder] = None,
                 lg: LandGraph = None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        # TODO: change value of WAY_LENGTH
        self.WAY_LENGTH = 10000000.0
        self.routing_mx: list[list[list[float, int, str]]] = None
        if holders is None:
            holders = []
        self.lg: LandGraph = lg
        self._holders: list[ResourceHolder] = holders

        # _ind2holder_id is required to match ResourceHolder's id to index in list of LangGraphNodes to work with routing_mx
        self._ind2holder_id: dict[int, str] = {self.lg.node2ind[holder.node]: holder.node.id for holder in
                                               self._holders}
        self.holder_id2resource_holder: dict[str, ResourceHolder] = {holder.node.id: holder for holder in self._holders}
        self.zone_config = zone_config

    def build_landscape(self):
        self.routing_mx = [[[self.WAY_LENGTH, -1, '0'] for j in range(self.lg.vertex_count)] for i in
                           range(self.lg.vertex_count)]
        self._build_routes()

    def get_sorted_holders(self, node_id: int) -> SortedList[list[tuple[tuple[float, int, str] | str]]]:
        """
        :param node_id: id of node in LandGraph's list of nodes
        :return: sorted list of holders' id by the length of way
        """
        holders = []
        for i in range(len(self.routing_mx)):
            if int(self.routing_mx[node_id][i][0]) != self.WAY_LENGTH:
                holders.append((self.routing_mx[node_id][i], self._ind2holder_id[i]))
        return SortedList(holders, key=lambda x: x[0][0])

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
        return [Road('road', edge) for edge in self.lg.roads]

    def get_all_resources(self) -> list[dict]:
        holders = {
            holder.id: {name: count for name, count in holder.node.resource_storage_unit.capacity.items()}.update(
                {vehicle.id: vehicle.get_sum_resources() for vehicle in holder.vehicles})
            for holder in self.holders}
        roads = {road.id: {'vehicles': road.vehicles}
                 for road in self.roads}
        platforms = {node.id: {name: count for name, count in node.resource_storage_unit.capacity.items()}
                     for node in self.platforms}

        resources = [holders, roads, platforms]
        return resources

    def _build_routes(self):

        def dijkstra(node_ind: int):
            priority_queue = [(0.0, node_ind)]
            visited = [False] * len(self.lg.nodes)

            while len(priority_queue) > 0:
                current_dist, current_v = heapq.heappop(priority_queue)
                visited[current_v] = True

                if current_dist > self.routing_mx[current_v][node_ind][0]:
                    continue

                for road in self.lg.nodes[current_v].roads:
                    dist: float = current_dist + road.weight
                    to_v = self.lg.node2ind[road.finish]
                    if visited[to_v]:
                        continue
                    if dist < self.routing_mx[node_ind][to_v][0]:
                        self.routing_mx[node_ind][to_v][0] = dist
                        self.routing_mx[node_ind][to_v][1] = \
                            self.routing_mx[current_v][node_ind][1]
                        self.routing_mx[node_ind][to_v][1] = current_v
                        self.routing_mx[node_ind][to_v][2] = road.id
                        heapq.heappush(priority_queue, (dist, to_v))

        node_indices = [self.lg.node2ind[node] for node in self.lg.nodes]
        for i in node_indices:
            dijkstra(i)


# TODO: delete
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
