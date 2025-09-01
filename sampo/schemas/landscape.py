"""Landscape resources and routing utilities.

Утилиты ландшафтных ресурсов и маршрутизации.
"""

import heapq
import math
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from sortedcontainers import SortedList

from sampo.schemas.landscape_graph import LandGraph, LandGraphNode, LandEdge
from sampo.schemas.resources import Material
from sampo.schemas.zones import ZoneConfiguration


class ResourceSupply(ABC):
    """Base entity that supplies resources.

    Базовая сущность, предоставляющая ресурсы.
    """

    def __init__(self, id: str, name: str):
        """Initialize supply with identifier and name.

        Инициализирует поставщика ресурсом идентификатором и именем.

        Args:
            id (str): unique identifier.
                Уникальный идентификатор.
            name (str): human-readable name.
                Человекочитаемое имя.
        """
        self.id = id
        self.name = name

    @abstractmethod
    def get_resources(self) -> list[tuple[str, int]]:
        """Return provided resources.

        Возвращает предоставляемые ресурсы.

        Returns:
            list[tuple[str, int]]: resource name and quantity pairs.
                Пары названия ресурса и количества.
        """
        ...


class Road(ResourceSupply):
    """Road segment between two nodes.

    Дорожный сегмент между двумя узлами.
    """

    def __init__(self, name: str, edge: LandEdge, speed: float = 5):
        """Initialize road based on graph edge.

        Инициализирует дорогу на основе ребра графа.

        Args:
            name (str): road name.
                Название дороги.
            edge (LandEdge): edge in the land graph.
                Ребро в графе ландшафта.
            speed (float): maximum speed on the road.
                Максимальная скорость на дороге.
        """
        super(Road, self).__init__(edge.id, name)
        self.vehicles = edge.bandwidth
        self.length = edge.weight
        self.speed = speed
        self.overcome_time = math.ceil(self.length / self.speed)
        self.edge = edge

    def get_resources(self) -> list[tuple[str, int]]:
        """Return road characteristics as resources.

        Возвращает характеристики дороги как ресурсы.

        Returns:
            list[tuple[str, int]]: resource name and value pairs.
                Пары названий и значений ресурсов.
        """
        return [('speed', self.speed), ('length', self.edge.weight), ('vehicles', self.vehicles)]


class Vehicle(ResourceSupply):
    """Transport vehicle with material capacity.

    Транспортное средство с грузоподъемностью.
    """

    def __init__(self, id: str, name: str, capacity: list[Material]):
        """Initialize vehicle with capacity.

        Инициализирует транспортное средство с вместимостью.

        Args:
            id (str): identifier of vehicle.
                Идентификатор транспорта.
            name (str): vehicle name.
                Название транспорта.
            capacity (list[Material]): carried materials.
                Перевозимые материалы.
        """
        super(Vehicle, self).__init__(id, name)
        self.capacity = capacity
        self.cost = 0.0
        self.volume = sum([mat.count for mat in self.capacity])

    @cached_property
    def resources(self) -> dict[str, int]:
        """Material quantities carried by the vehicle.

        Количество материалов, перевозимых транспортом.
        """
        return {mat.name: mat.count for mat in self.capacity}

    def get_resources(self) -> list[tuple[str, int]]:
        """Return vehicle capacity as resources.

        Возвращает вместимость транспорта в виде ресурсов.

        Returns:
            list[tuple[str, int]]: material name and count pairs.
                Пары названий материалов и количества.
        """
        return [(mat.name, mat.count) for mat in self.capacity]


class ResourceHolder(ResourceSupply):
    """Storage node that owns vehicles.

    Узел хранения, владеющий транспортом.
    """

    def __init__(self, id: str, name: str,
                 vehicles: list[Vehicle] = None,
                 node: LandGraphNode = None):
        """Initialize resource holder.

        Инициализирует держателя ресурсов.

        Args:
            id (str): identifier of holder.
                Идентификатор держателя.
            name (str): holder name.
                Название держателя.
            vehicles (list[Vehicle] | None): available vehicles.
                Доступные транспортные средства.
            node (LandGraphNode | None): associated land graph node.
                Связанный узел ландшафтного графа.
        """
        # let ids of two objects will be similar to make simpler matching ResourceHolder to node in LandGraph
        super(ResourceHolder, self).__init__(id, name)
        self.node_id = node.id
        self.vehicles = vehicles
        self.node = node

    def get_resources(self) -> list[tuple[str, int]]:
        """Return resources stored at the node.

        Возвращает ресурсы, хранящиеся в узле.

        Returns:
            list[tuple[str, int]]: resource name and quantity pairs.
                Пары названий ресурсов и количеств.
        """
        return [(name, count) for name, count in self.node.resource_storage_unit.capacity.items()]


class LandscapeConfiguration:
    """Configuration of resource holders and routes.

    Конфигурация держателей ресурсов и маршрутов.
    """

    def __init__(self,
                 holders: list[ResourceHolder] = None,
                 lg: LandGraph = None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        self.WAY_LENGTH = np.inf
        self.dist_mx: list[list[float]] = None
        self.path_mx: np.array = None
        self.road_mx: list[list[str]] = None
        if holders is None:
            holders = []
        self.lg: LandGraph = lg
        self._holders: list[ResourceHolder] = holders

        # _ind2holder_id is required to match ResourceHolder's id to index in list of LangGraphNodes to work with routing_mx
        self.ind2holder_node_id: dict[int, str] = {self.lg.node2ind[holder.node]: holder.node.id for holder in
                                                   self._holders}
        self.holder_node_id2resource_holder: dict[str, ResourceHolder] = {holder.node.id: holder for holder in self._holders}
        self.zone_config = zone_config
        if self.lg is None:
            return
        self.works2platform = {}
        for platform in self.platforms:
            for work in platform.works:
                self.works2platform[work] = platform
        # self.works2platform: dict['GraphNode', LandGraphNode] = {work: platform for platform in self.platforms
        #                                                          for work in platform.works}
        self._build_routes()
        self._node2ind = self.lg.node2ind

    def get_sorted_holders(self, node: LandGraphNode) -> SortedList[list[tuple[float, str]]]:
        """Return holders sorted by distance from node.

        Возвращает держателей, отсортированных по расстоянию от узла.

        Args:
            node (LandGraphNode): node of interest.
                Исходный узел.

        Returns:
            SortedList[list[tuple[float, str]]]: pairs of distance and holder id.
                Пары расстояния и идентификатора держателя.
        """
        holders = []
        node_ind = self._node2ind[node]
        for i in self.ind2holder_node_id.keys():
            if self.dist_mx[node_ind][i] != self.WAY_LENGTH:
                holders.append((self.dist_mx[node_ind][i], self.ind2holder_node_id[i]))
        return SortedList(holders, key=lambda x: x[0])

    def get_route(self, from_node: LandGraphNode, to_node: LandGraphNode) -> list[str]:
        """Return road IDs forming the shortest route.

        Возвращает идентификаторы дорог, составляющих кратчайший маршрут.

        Args:
            from_node (LandGraphNode): start node.
                Начальный узел.
            to_node (LandGraphNode): destination node.
                Конечный узел.

        Returns:
            list[str]: identifiers of roads along the path.
                Идентификаторы дорог вдоль пути.
        """
        from_ind = self._node2ind[from_node]
        to_ind = self._node2ind[to_node]

        path = [to_ind]
        to = to_ind
        while self.path_mx[from_ind][to] != from_ind:
            path.append(self.path_mx[from_ind][to])
            to = self.path_mx[from_ind][to]
        path.append(from_ind)

        return [self.road_mx[path[v - 1]][path[v]] for v in range(len(path) - 1, 0, -1)]

    def _dijkstra(self, node_ind: int, distances: list[float], path_mx: np.ndarray, roads_available_set: set[str]) \
            -> np.ndarray:
        prior_queue = [(.0, node_ind)]
        distances[node_ind] = 0

        while prior_queue:
            dist, v = heapq.heappop(prior_queue)

            if dist > distances[v]:
                continue

            for road in self.lg.nodes[v].roads:
                if road.id not in roads_available_set:
                    continue
                d = dist + road.weight
                finish_ind = self._node2ind[road.finish]
                if d < distances[finish_ind]:
                    distances[finish_ind] = d
                    path_mx[node_ind][finish_ind] = v
                    heapq.heappush(prior_queue, (d, finish_ind))
        return path_mx

    def construct_route(self, from_node: LandGraphNode, to_node: LandGraphNode,
                        roads_available: list[Road]) -> list[str]:
        """Construct route using subset of available roads.

        Строит маршрут, используя доступное множество дорог.

        Args:
            from_node (LandGraphNode): start node.
                Начальный узел.
            to_node (LandGraphNode): destination node.
                Конечный узел.
            roads_available (list[Road]): roads that may be used.
                Доступные для использования дороги.

        Returns:
            list[str]: road IDs if route exists, otherwise empty list.
                Идентификаторы дорог, если маршрут существует, иначе пустой список.
        """

        adj_matrix = self.lg.adj_matrix.copy()
        distances = [self.WAY_LENGTH] * self.lg.vertex_count
        path_mx = np.full((self.lg.vertex_count, self.lg.vertex_count), -1)
        roads_available_set = set(road.id for road in roads_available)
        for v in range(self.lg.vertex_count):
            for u in range(self.lg.vertex_count):
                if v == u:
                    path_mx[v][u] = 0
                elif adj_matrix[v][u] != np.inf and self.road_mx[v][u] in roads_available_set:
                    path_mx[v][u] = v

        from_ind = self._node2ind[from_node]
        to_ind = self._node2ind[to_node]
        path_mx = self._dijkstra(to_ind, distances, path_mx, roads_available_set)

        if path_mx[to_ind][from_ind] == -1:
            return []

        fr = from_ind
        path = [from_ind]
        while path_mx[to_ind][fr] != to_ind:
            path.append(path_mx[to_ind][fr])
            fr = path_mx[to_ind][fr]
        path.append(to_ind)

        return [self.road_mx[path[v]][path[v + 1]] for v in range(len(path) - 1)]

    @cached_property
    def holders(self) -> list[ResourceHolder]:
        """All resource holders.

        Все держатели ресурсов.
        """
        return self._holders

    @cached_property
    def platforms(self) -> list[LandGraphNode]:
        """Nodes without assigned resource holders.

        Узлы без назначенных держателей ресурсов.
        """
        if self.lg is None:
            return []
        platform_ids = set([node.id for node in self.lg.nodes]).difference(
            set([holder.node_id for holder in self._holders]))
        return [node for node in self.lg.nodes if node.id in platform_ids]

    @cached_property
    def roads(self) -> list[Road]:
        """All roads in the landscape graph.

        Все дороги в графе ландшафта.
        """
        return [Road(f'road_{i}', edge) for i, edge in enumerate(self.lg.edges)]

    def get_all_resources(self) -> list[dict[str, dict[str, int]]]:
        """Collect resources of holders and roads.

        Собирает ресурсы держателей и дорог.

        Returns:
            list[dict[str, dict[str, int]]]: resource information for holders and roads.
                Информация о ресурсах держателей и дорог.
        """

        def merge_dicts(a, b):
            c = a.copy()
            c.update(b)
            return c

        if self.lg is None:
            return []

        holders = {
            holder.id: merge_dicts({name: count for name, count in holder.get_resources()},
                                   {'vehicles': len(holder.vehicles)})
            for holder in self.holders}
        roads = {road.id: {'vehicles': road.vehicles}
                 for road in self.roads}

        resources = [holders, roads]
        return resources

    def get_platforms_resources(self) -> dict[str, dict[str, int]]:
        """Resources available at platforms.

        Ресурсы, доступные на площадках.

        Returns:
            dict[str, dict[str, int]]: mapping of platform id to its resources.
                Отображение id площадки на ее ресурсы.
        """
        return {node.id: {name: count for name, count in node.resource_storage_unit.capacity.items()}
                for node in self.platforms}

    def _build_routes(self):
        """Precompute routing matrices.

        Предварительно вычисляет матрицы маршрутов.
        """
        count = self.lg.vertex_count
        dist_mx = self.lg.adj_matrix.copy()
        path_mx: np.array = np.full((count, count), -1)
        road_mx: list[list[str]] = [['-1' for j in range(count)] for i in range(count)]

        for v in range(count):
            for u in range(count):
                if v == u:
                    path_mx[v][u] = 0
                elif dist_mx[v][u] != np.inf:
                    path_mx[v][u] = v
                    for road in self.lg.nodes[v].roads:
                        if self.lg.node2ind[road.finish] == u:
                            road_mx[v][u] = road.id
                else:
                    path_mx[v][u] = -1

        for i in range(self.lg.vertex_count):
            for u in range(self.lg.vertex_count):
                for v in range(self.lg.vertex_count):
                    if (dist_mx[u][i] != np.inf and dist_mx[u][i] != 0
                            and dist_mx[i][v] != np.inf and dist_mx[i][v] != 0
                            and dist_mx[u][i] + dist_mx[i][v] < dist_mx[u][v]):
                        dist_mx[u][v] = dist_mx[u][i] + dist_mx[i][v]
                        path_mx[u][v] = path_mx[i][v]

        self.dist_mx = dist_mx
        self.path_mx = path_mx
        self.road_mx = road_mx


class MaterialDelivery:
    """Schedule of material deliveries for a work.

    Расписание поставок материалов для работы.
    """

    def __init__(self, work_id: str):
        """Initialize delivery schedule for work.

        Инициализирует расписание поставок для работы.

        Args:
            work_id (str): identifier of work.
                Идентификатор работы.
        """
        self.id = work_id
        self.delivery = {}

    def add_delivery(self, name: str, count: int, start_time: 'Time', finish_time: 'Time', from_holder: str):
        """Record single material delivery.

        Записывает одну поставку материала.

        Args:
            name (str): material name.
                Название материала.
            count (int): delivered amount.
                Доставляемое количество.
            start_time (Time): start time.
                Время начала.
            finish_time (Time): finish time.
                Время окончания.
            from_holder (str): source holder id.
                Идентификатор держателя-источника.
        """
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            material_delivery = []
            self.delivery[name] = material_delivery
        material_delivery.append((count, start_time, finish_time, from_holder))

    def add_deliveries(self, name: str, deliveries: list[tuple[int, 'Time', 'Time', str]]):
        """Record multiple material deliveries.

        Записывает несколько поставок материала.

        Args:
            name (str): material name.
                Название материала.
            deliveries (list[tuple[int, Time, Time, str]]): list of deliveries.
                Список поставок.
        """
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            self.delivery[name] = deliveries
        else:
            material_delivery.extend(deliveries)
