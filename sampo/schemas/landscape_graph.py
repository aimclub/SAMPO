import uuid
"""Landscape graph structures.

Структуры графа ландшафта.
"""

import uuid
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from sampo.schemas.exceptions import NoAvailableResources


@dataclass
class LandEdge:
    """Connection between two vertices of a transport graph.

    Соединение между двумя вершинами транспортного графа.

    Attributes:
        id (str): identifier of the edge.
            Идентификатор ребра.
        start (LandGraphNode): start node.
            Начальная вершина.
        finish (LandGraphNode): finish node.
            Конечная вершина.
        weight (float): length of the edge.
            Длина ребра.
        bandwidth (int): number of vehicles per hour.
            Количество транспортных средств в час.
    """
    id: str
    start: 'LandGraphNode'
    finish: 'LandGraphNode'
    weight: float
    bandwidth: int


class ResourceStorageUnit:
    """Resource storage for a land graph node.

    Хранилище ресурсов для узла ландшафтного графа.
    """

    def __init__(self, capacity: dict[str, int] | None = None):
        """Initialize storage with capacity.

        Инициализирует хранилище с вместимостью.

        Args:
            capacity (dict[str, int] | None): maximum values for materials.
                Максимальные значения для материалов.
        """
        if capacity is None:
            capacity = {}
        self._capacity = capacity

    @cached_property
    def capacity(self) -> dict[str, int]:
        """Maximum available amount of each resource.

        Максимальное доступное количество каждого ресурса.
        """
        return self._capacity


class LandGraphNode:
    """Participant of the landscape transport network.

    Участник транспортной сети ландшафта.
    """

    def __init__(self,
                 id: str = str(uuid.uuid4()),
                 name: str = 'platform',
                 resource_storage_unit: ResourceStorageUnit = ResourceStorageUnit(),
                 neighbour_nodes: list[tuple['LandGraphNode', float, int]] | None = None,
                 works: list['GraphNode'] | Any = None):
        """Initialize landscape graph node.

        Инициализирует узел ландшафтного графа.

        Args:
            id (str): node identifier.
                Идентификатор узла.
            name (str): node name.
                Название узла.
            resource_storage_unit (ResourceStorageUnit): storage information.
                Информация о хранилище.
            neighbour_nodes (list[tuple[LandGraphNode, float, int]] | None):
                neighbours with distance and bandwidth.
                Соседи с расстоянием и пропускной способностью.
            works (list[GraphNode] | Any | None): works assigned to the node.
                Работы, назначенные узлу.
        """
        self.id = id
        self.name = name
        if not (neighbour_nodes is None):
            self.add_neighbours(neighbour_nodes)
        self._roads: list[LandEdge] = []
        self.nodes: list['GraphNode'] = []
        if works is not None:
            self.nodes.extend(works) if isinstance(works, list) else self.nodes.append(works)
        self.resource_storage_unit = resource_storage_unit

    def __hash__(self):
        return hash(self.id)

    @cached_property
    def neighbours(self) -> list['LandGraphNode']:
        """Adjacent nodes connected by roads.

        Узлы-соседи, соединенные дорогами.
        """
        if self._roads:
            return [neighbour_edge.finish for neighbour_edge in self._roads]
        return []

    @cached_property
    def roads(self) -> list[LandEdge]:
        """Outgoing roads from the node.

        Исходящие из узла дороги.

        Raises:
            NoAvailableResources: if no roads are connected.
                Если нет соединенных дорог.
        """
        if self._roads:
            return self._roads
        raise NoAvailableResources('There are no roads in land graph')

    def add_works(self, nodes: list['GraphNode'] | Any):
        """Attach works to the node.

        Присоединяет работы к узлу.

        Args:
            nodes (list[GraphNode] | GraphNode): works to attach.
                Работы для присоединения.
        """
        if isinstance(nodes, list):
            self.nodes.extend(nodes)
        else:
            self.nodes.append(nodes)

    @cached_property
    def works(self) -> list['GraphNode']:
        """Works associated with the node.

        Работы, связанные с узлом.
        """
        return self.nodes

    def add_neighbours(self, neighbour_nodes: list[tuple['LandGraphNode', float, int]] | tuple['LandGraphNode', float, int]):
        """Connect node with neighbours.

        Соединяет узел с соседями.

        Args:
            neighbour_nodes (list | tuple): neighbours with distance and bandwidth.
                Соседи с расстоянием и пропускной способностью.
        """
        if isinstance(neighbour_nodes, list):
            for neighbour, length, bandwidth in neighbour_nodes:
                road_id = str(uuid.uuid4())
                self._roads.append(LandEdge(road_id, self, neighbour, length, bandwidth))
                neighbour._roads.append(LandEdge(road_id, neighbour, self, length, bandwidth))
        else:
            neighbour, length, bandwidth = neighbour_nodes
            road_id = str(uuid.uuid4())
            self._roads.append(LandEdge(road_id, self, neighbour, length, bandwidth))
            neighbour._roads.append(LandEdge(road_id, neighbour, self, length, bandwidth))

@dataclass
class LandGraph:
    """Graph representing the landscape transport network.

    Граф, представляющий транспортную сеть ландшафта.
    """

    nodes: list[LandGraphNode] = None
    adj_matrix: np.array = None
    node2ind: dict[LandGraphNode, int] = None
    id2ind: dict[str, int] = None
    vertex_count: int = None

    def __post_init__(self) -> None:
        self.reinit()

    def reinit(self):
        """Rebuild adjacency matrix and indices.

        Перестраивает матрицу смежности и индексы.
        """
        adj_matrix, node2ind, id2ind = self._to_adj_matrix()
        object.__setattr__(self, 'adj_matrix', adj_matrix)
        object.__setattr__(self, 'node2ind', node2ind)
        object.__setattr__(self, 'id2ind', id2ind)
        object.__setattr__(self, 'vertex_count', len(node2ind))

    @cached_property
    def edges(self) -> list['LandEdge']:
        """Unique list of edges in the graph.

        Уникальный список ребер графа.
        """
        road_ids = set()
        roads = []

        for node in self.nodes:
            for road in node.roads:
                if road.id not in road_ids:
                    road_ids.add(road.id)
                    roads.append(road)

        return roads

    def _to_adj_matrix(self) -> tuple[np.array, dict[LandGraphNode, int], dict[str, int]]:
        """Convert graph to adjacency matrix.

        Преобразует граф в матрицу смежности.

        Returns:
            tuple[np.array, dict[LandGraphNode, int], dict[str, int]]: adjacency matrix,
                node-to-index and id-to-index mappings.
            tuple[np.array, dict[LandGraphNode, int], dict[str, int]]: матрица смежности,
                отображения узел-индекс и id-индекс.
        """
        node2ind: dict[LandGraphNode, int] = {
            v: i for i, v in enumerate(self.nodes)
        }
        id2ind = {
            v.id: i for i, v in enumerate(self.nodes)
        }
        adj_mtrx = np.full((len(node2ind), len(node2ind)), np.inf)
        for v, i in node2ind.items():
            for child in v.roads:
                c_i = node2ind[child.finish]
                adj_mtrx[i, c_i] = child.weight

        return adj_mtrx, node2ind, id2ind
