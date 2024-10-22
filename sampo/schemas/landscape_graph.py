import uuid
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from sampo.schemas.exceptions import NoAvailableResources


@dataclass
class LandEdge:
    """
    A representation of the connection between two vertices of a transport graph
    (e.x., the edge between platform and warehouse)
    """
    id: str
    start: 'LandGraphNode'
    finish: 'LandGraphNode'
    weight: float
    bandwidth: int


class ResourceStorageUnit:
    def __init__(self, capacity: dict[str, int] | None = None):
        """
        Represents the resource storage of a land graph node
        :param capacity: list of the maximum values of each material
        """
        if capacity is None:
            capacity = {}
        self._capacity = capacity

    @cached_property
    def capacity(self) -> dict[str, int]:
        return self._capacity


class LandGraphNode:
    def __init__(self,
                 id: str = str(uuid.uuid4()),
                 name: str = 'platform',
                 resource_storage_unit: ResourceStorageUnit = ResourceStorageUnit(),
                 neighbour_nodes: list[tuple['LandGraphNode', float, int]] | None = None,
                 works: list['GraphNode'] | Any = None):
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
        self._roads: list[LandEdge] = []
        self.nodes: list['GraphNode'] = []
        if works is not None:
            self.nodes.extend(works) if isinstance(works, list) else self.nodes.append(works)
        self.resource_storage_unit = resource_storage_unit

    def __hash__(self):
        return hash(self.id)

    @cached_property
    def neighbours(self) -> list['LandGraphNode']:
        if self._roads:
            return [neighbour_edge.finish for neighbour_edge in self._roads]
        return []

    @cached_property
    def roads(self) -> list[LandEdge]:
        if self._roads:
            return self._roads
        raise NoAvailableResources('There are no roads in land graph')

    def add_works(self, nodes: list['GraphNode'] | Any):
        if isinstance(nodes, list):
            self.nodes.extend(nodes)
        else:
            self.nodes.append(nodes)

    @cached_property
    def works(self) -> list['GraphNode']:
        return self.nodes

    def add_neighbours(self, neighbour_nodes: list[tuple['LandGraphNode', float, int]] | tuple['LandGraphNode', float, int]):
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
    nodes: list[LandGraphNode] = None
    adj_matrix: np.array = None
    node2ind: dict[LandGraphNode, int] = None
    id2ind: dict[str, int] = None
    vertex_count: int = None

    def __post_init__(self) -> None:
        self.reinit()

    def reinit(self):
        adj_matrix, node2ind, id2ind = self._to_adj_matrix()
        object.__setattr__(self, 'adj_matrix', adj_matrix)
        object.__setattr__(self, 'node2ind', node2ind)
        object.__setattr__(self, 'id2ind', id2ind)
        object.__setattr__(self, 'vertex_count', len(node2ind))

    @cached_property
    def edges(self) -> list['LandEdge']:
        # TODO: to do
        road_ids = set()
        roads = []

        for node in self.nodes:
            for road in node.roads:
                if road.id not in road_ids:
                    road_ids.add(road.id)
                    roads.append(road)

        return roads

    def _to_adj_matrix(self) -> tuple[np.array, dict[LandGraphNode, int], dict[str, int]]:
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
