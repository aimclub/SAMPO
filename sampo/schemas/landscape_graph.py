from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.sparse import dok_matrix

from sampo.schemas.exceptions import NoAvailableResources
from sampo.schemas.resources import Material


@dataclass
class Road:
    """
    A representation of the connection between two vertices of a transport graph
    (e.x., the edge between platform and warehouse)
    """
    start: 'LandGraphNode'
    finish: 'LandGraphNode'
    weight: float


class ResourceStorageUnit:
    def __init__(self, capacity: list[Material]):
        """
        Represents the resource storage of a land graph node
        :param capacity: list of the maximum values of each material
        """
        self._capacity = capacity

    def get_capacity(self) -> list[Material]:
        return self._capacity


class LandGraphNode:
    def __init__(self,
                 id: str,
                 name: str,
                 resource_storage_unit: ResourceStorageUnit,
                 neighbour_nodes: list[tuple['LandGraphNode', float]] | None = None):
        """
        Represents the participant of landscape transport network

        :param neighbour_nodes: parent nodes list that saves parent itself and the weight of edge
        :param resource_storage_unit: object that saves info about the resources in the current node
        (in other words platform)
        """
        self.id = id
        self.name = name
        if not (neighbour_nodes is None):
            self.add_neighbours(neighbour_nodes)
        self._roads: list[Road] = []
        self.nodes: list['GraphNode'] = []
        self.resource_storage_unit = resource_storage_unit

    @cached_property
    def neighbours(self) -> list['LandGraphNode']:
        if self._roads:
            return [neighbour for neighbour, weight in self._roads]
        raise NoAvailableResources('There is no roads in land graph')

    @cached_property
    def roads(self) -> list[Road]:
        if self._roads:
            return self._roads
        raise NoAvailableResources('There are no roads in land graph')

    def add_works(self, nodes: list['GraphNode']):
        self.nodes = nodes

    @cached_property
    def works(self) -> list['GraphNode']:
        return self.nodes

    def add_neighbours(self, neighbour_nodes: list[tuple['LandGraphNode', float]]):
        [self._roads.append(Road(self, p, length)) for p, length in neighbour_nodes]


@dataclass
class LandGraph:
    nodes: list[LandGraphNode] = None
    adj_matrix: dok_matrix = None
    node2ind: dict[LandGraphNode, int] = None
    vertex_count: int = None

    def __post_init__(self) -> None:
        self.reinit()

    def reinit(self):
        adj_matrix, node2ind = self._to_adj_matrix()
        object.__setattr__(self, 'adj_matrix', adj_matrix)
        object.__setattr__(self, 'node2ind', node2ind)
        object.__setattr__(self, 'vertex_count', len(node2ind))

    @cached_property
    def roads(self) -> list['Road']:
        roads = []
        for i in range(self.vertex_count):
            for j in range(self.vertex_count):
                if self.adj_matrix[i][j] > 0 and i != j:
                    roads.append(Road(self.nodes[i], self.nodes[j], self.adj_matrix[i][j]))
        return roads

    def _to_adj_matrix(self) -> tuple[dok_matrix, dict[LandGraphNode, int]]:
        node2ind: dict[LandGraphNode, int] = {
            v: i for i, v in enumerate(self.nodes)
        }
        adj_mtrx = dok_matrix((len(node2ind), len(node2ind)), dtype=np.short)
        for v, i in node2ind.items():
            for child in v.roads:
                c_i = node2ind[child.finish]
                adj_mtrx[i, c_i] = child.weight

        return adj_mtrx, node2ind
