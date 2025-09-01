"""Resources and routing on the physical landscape.

Ресурсы и маршрутизация на физическом ландшафте.
"""

import heapq
import math
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from sortedcontainers import SortedList

from sampo.schemas.landscape_graph import LandEdge, LandGraph, LandGraphNode
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.zones import ZoneConfiguration


class ResourceSupply(ABC):
    """Base entity providing resources.

    Базовая сущность, предоставляющая ресурсы.
    """
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    @abstractmethod
    def get_resources(self) -> list[tuple[str, int]]:
        """Return provided resources.

        Возвращает предоставляемые ресурсы.
        """

        ...


class Road(ResourceSupply):
    """Road represented by an edge of the land graph.

    Дорога, представленная ребром ландшафтного графа.
    """

    def __init__(self, name: str, edge: LandEdge, speed: float = 5):
        """Initialize road parameters.

        Инициализирует параметры дороги.

        Args:
            name (str): Road name. / Название дороги.
            edge (LandEdge): Edge in land graph. / Ребро графа.
            speed (float): Max speed on road. / Максимальная скорость.
        """
        super(Road, self).__init__(edge.id, name)
        self.vehicles = edge.bandwidth
        self.length = edge.weight
        self.speed = speed
        self.overcome_time = math.ceil(self.length / self.speed)
        self.edge = edge

    def get_resources(self) -> list[tuple[str, int]]:
        """Return road properties as resources.

        Возвращает параметры дороги как ресурсы.
        """
        return [('speed', self.speed), ('length', self.edge.weight),
                ('vehicles', self.vehicles)]


class Vehicle(ResourceSupply):
    """Transport vehicle capable of carrying materials.

    Транспортное средство, способное перевозить материалы.
    """

    def __init__(self, id: str, name: str, capacity: list[Material]):
        """Initialize vehicle parameters.

        Инициализирует параметры транспортного средства.

        Args:
            id (str): Vehicle identifier. / Идентификатор.
            name (str): Vehicle name. / Название.
            capacity (list[Material]): Load capacity. / Грузоподъёмность.
        """
        super(Vehicle, self).__init__(id, name)
        self.capacity = capacity
        self.cost = 0.0
        self.volume = sum([mat.count for mat in self.capacity])

    @cached_property
    def resources(self) -> dict[str, int]:
        """Mapping of material names to counts.

        Отображение названий материалов в их количество.
        """
        return {mat.name: mat.count for mat in self.capacity}

    def get_resources(self) -> list[tuple[str, int]]:
        """Return carried materials as resources.

        Возвращает перевозимые материалы как ресурсы.
        """
        return [(mat.name, mat.count) for mat in self.capacity]


class ResourceHolder(ResourceSupply):
    """Node that stores vehicles and materials.

    Узел, хранящий технику и материалы.
    """

    def __init__(self, id: str, name: str, vehicles: list[Vehicle] | None = None,
                 node: LandGraphNode | None = None):
        """Initialize holder parameters.

        Инициализирует параметры склада.

        Args:
            id (str): Holder identifier. / Идентификатор склада.
            name (str): Holder name. / Название.
            vehicles (list[Vehicle] | None): Available vehicles. /
                Доступные транспортные средства.
            node (LandGraphNode | None): Associated node. / Связанный узел.
        """
        # let ids of two objects will be similar to make simpler matching ResourceHolder to node in LandGraph
        super(ResourceHolder, self).__init__(id, name)
        self.node_id = node.id
        self.vehicles = vehicles
        self.node = node

    def get_resources(self) -> list[tuple[str, int]]:
        """Return stored materials as resources.

        Возвращает хранимые материалы как ресурсы.
        """
        return [(name, count) for name, count in
                self.node.resource_storage_unit.capacity.items()]


class LandscapeConfiguration:
    """Configuration for routing resources across the landscape.

    Конфигурация маршрутизации ресурсов по ландшафту.
    """

    def __init__(self, holders: list[ResourceHolder] | None = None,
                 lg: LandGraph | None = None,
                 zone_config: ZoneConfiguration = ZoneConfiguration()):
        """Set up landscape configuration.

        Настраивает конфигурацию ландшафта.

        Args:
            holders (list[ResourceHolder] | None): Resource holders. /
                Складские узлы.
            lg (LandGraph | None): Landscape graph. / Ландшафтный граф.
            zone_config (ZoneConfiguration): Zone settings. / Настройки зон.
        """
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
        """Return holders sorted by path length to ``node``.

        Возвращает склады, отсортированные по длине пути до ``node``.

        Args:
            node (LandGraphNode): Node in land graph. / Узел графа.

        Returns:
            SortedList[list[tuple[float, str]]]: Sorted holders. /
            Отсортированные склады.
        """
        holders = []
        node_ind = self._node2ind[node]
        for i in self.ind2holder_node_id.keys():
            if self.dist_mx[node_ind][i] != self.WAY_LENGTH:
                holders.append((self.dist_mx[node_ind][i], self.ind2holder_node_id[i]))
        return SortedList(holders, key=lambda x: x[0])

    def get_route(self, from_node: LandGraphNode, to_node: LandGraphNode) -> list[str]:
        """Get list of road identifiers forming the route.

        Получить список идентификаторов дорог, составляющих маршрут.
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
        """Construct route using only available roads.

        Построить маршрут, используя только доступные дороги.

        Args:
            from_node (LandGraphNode): Start node. / Узел начала.
            to_node (LandGraphNode): End node. / Узел окончания.
            roads_available (list[Road]): Available roads. / Доступные дороги.

        Returns:
            list[str]: Road identifiers or empty list. / Идентификаторы дорог
            или пустой список.
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
        return self._holders

    @cached_property
    def platforms(self) -> list[LandGraphNode]:
        if self.lg is None:
            return []
        platform_ids = set([node.id for node in self.lg.nodes]).difference(
            set([holder.node_id for holder in self._holders]))
        return [node for node in self.lg.nodes if node.id in platform_ids]

    @cached_property
    def roads(self) -> list[Road]:
        return [Road(f'road_{i}', edge) for i, edge in enumerate(self.lg.edges)]

    def get_all_resources(self) -> list[dict[str, dict[str, int]]]:
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
        return {node.id: {name: count for name, count in node.resource_storage_unit.capacity.items()}
                for node in self.platforms}

    def _build_routes(self):
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
    """Information about material deliveries for a work.

    Сведения о поставках материалов для работы.
    """

    def __init__(self, work_id: str):
        """Create delivery storage for work.

        Создаёт хранилище поставок для работы.

        Args:
            work_id (str): Work identifier. / Идентификатор работы.
        """
        self.id = work_id
        self.delivery = {}

    def add_delivery(self, name: str, count: int, start_time: Time,
                     finish_time: Time, from_holder: str) -> None:
        """Add single delivery record.

        Добавляет запись о поставке.

        Args:
            name (str): Material name. / Название материала.
            count (int): Delivered amount. / Количество.
            start_time (Time): Start time. / Время начала.
            finish_time (Time): Finish time. / Время окончания.
            from_holder (str): Source holder. / Источник.
        """
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            material_delivery = []
            self.delivery[name] = material_delivery
        material_delivery.append((count, start_time, finish_time, from_holder))

    def add_deliveries(self, name: str,
                       deliveries: list[tuple[int, Time, Time, str]]) -> None:
        """Add multiple delivery records.

        Добавляет несколько записей о поставках.

        Args:
            name (str): Material name. / Название материала.
            deliveries (list[tuple[int, Time, Time, str]]): Delivery list. /
                Список поставок.
        """
        material_delivery = self.delivery.get(name, None)
        if material_delivery is None:
            self.delivery[name] = deliveries
        else:
            material_delivery.extend(deliveries)
