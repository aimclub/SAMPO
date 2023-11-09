from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property, cache
from typing import Optional

import numpy as np
from scipy.sparse import dok_matrix

from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.serializable import JSONSerializable, T, JS
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit


class EdgeType(Enum):
    """
    Class to define a certain type of edge in graph
    """
    InseparableFinishStart = 'IFS'
    LagFinishStart = 'FFS'
    StartStart = 'SS'
    FinishFinish = 'FF'
    FinishStart = 'FS'

    @staticmethod
    def is_dependency(edge) -> bool:
        if edge == '-1':  # ... no comments
            return True
        if isinstance(edge, EdgeType):
            edge = edge.value
        return edge in ('FS', 'IFS', 'FFS')


@dataclass
class GraphEdge:
    """
    The edge of graph with start and finish vertexes
    """
    start: 'GraphNode'
    finish: 'GraphNode'
    lag: float | None = 0
    type: EdgeType | None = None


class GraphNode(JSONSerializable['GraphNode']):
    """
    Class to describe Node in graph
    """

    def __init__(self, work_unit: WorkUnit,
                 parent_works: list['GraphNode'] | list[tuple['GraphNode', float, EdgeType]]):
        self._work_unit = work_unit
        self._parent_edges = []
        self.add_parents(parent_works)
        self._children_edges = []

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return self.id

    def __getstate__(self):
        # custom method to avoid calling __hash__() on GraphNode objects
        return self._work_unit._serialize(), \
            [(e.start.id, e.lag, e.type.value) for e in self._parent_edges]

    def __setstate__(self, state):
        # custom method to avoid calling __hash__() on GraphNode objects
        s_work_unit, s_parent_edges = state
        self.__init__(WorkUnit._deserialize(s_work_unit),
                      s_parent_edges)
        # self._work_unit = representation['work_unit']
        # self._parent_edges = [GraphEdge(*e) for e in representation['parent_edges']]

    def _serialize(self) -> T:
        return {
            'work_unit': self._work_unit._serialize(),
            'parent_edges': [(e.start.id, e.lag, e.type.value) for e in self._parent_edges],
            # 'child_edges': [(e.finish.work_unit.id, e.lag, e.type.value) for e in self._children_edges]
        }

    @classmethod
    def _deserialize(cls, representation: T) -> dict:
        representation['work_unit'] = WorkUnit._deserialize(representation['work_unit'])
        representation['parent_edges'] = [(e[0], e[1], EdgeType(e[2])) for e in representation['parent_edges']]
        # representation['child_edges'] = [(e[0], e[1], EdgeType(e[2])) for e in representation['child_edges']]
        return representation

    def update_work_unit(self, work_unit: WorkUnit) -> None:
        self._work_unit = work_unit

    def add_parents(self, parent_works: list['GraphNode'] or list[tuple['GraphNode', float, EdgeType]]) -> None:
        """
        Two-sided linking of successors and predecessors

        :param parent_works: list of parent works
        """
        edges: list[GraphEdge] = []
        if parent_works:
            if isinstance(parent_works[0], GraphNode):
                edges = [GraphEdge(p, self, 1, EdgeType.FinishStart) for p in parent_works]
            elif isinstance(parent_works[0], tuple):
                edges = [GraphEdge(p, self, lag, edge_type) for p, lag, edge_type in parent_works]

        for edge, parent in zip(edges, parent_works):
            parent: GraphNode = parent[0] if isinstance(parent, tuple) else parent
            parent._add_child_edge(edge)
            parent.invalidate_children_cache()
        self._parent_edges += edges
        self.invalidate_parents_cache()

    def invalidate_parents_cache(self):
        self.__dict__.pop('parents', None)
        self.__dict__.pop('parents_set', None)
        self.__dict__.pop('inseparable_parent', None)
        self.__dict__.pop('inseparable_son', None)
        self.__dict__.pop('get_inseparable_chain', None)

    def invalidate_children_cache(self):
        self.__dict__.pop('children', None)
        self.__dict__.pop('children_set', None)
        self.__dict__.pop('inseparable_parent', None)
        self.__dict__.pop('inseparable_son', None)
        self.__dict__.pop('get_inseparable_chain', None)

    def is_inseparable_parent(self) -> bool:
        return self.inseparable_son is not None

    def is_inseparable_son(self) -> bool:
        return self.inseparable_parent is not None

    def traverse_children(self, topologically: bool = False):
        """
        DFS from current vertex to down
        :param topologically: is DFS need to go in topologically way
        :return:
        """
        visited_vertexes = set()
        vertexes_to_visit = deque([self])
        while len(vertexes_to_visit) > 0:
            v = vertexes_to_visit.popleft()
            if topologically and any(p.start not in visited_vertexes for p in v._parent_edges):
                vertexes_to_visit.append(v)
                continue
            if v not in visited_vertexes:
                visited_vertexes.add(v)
                vertexes_to_visit.extend([p.finish for p in v._children_edges])
                yield v

    @cached_property
    def inseparable_son(self) -> Optional['GraphNode']:
        """
        Return inseparable son (amount of inseparable sons at most 1)
        :return: inseparable son
        """
        inseparable_children = [x.finish for x in self._children_edges
                                if x.type == EdgeType.InseparableFinishStart]
        return inseparable_children[0] if inseparable_children else None

    @cached_property
    def inseparable_parent(self) -> Optional['GraphNode']:
        """
        Return predecessor of current vertex in inseparable chain
        :return: inseparable parent
        """
        inseparable_parents = [x.start for x in self._parent_edges if x.type == EdgeType.InseparableFinishStart]
        return inseparable_parents[0] if inseparable_parents else None

    @cached_property
    def parents(self) -> list['GraphNode']:
        """
        Return list of predecessors of current vertex
        :return: list of parents
        """
        return [edge.start for edge in self.edges_to if EdgeType.is_dependency(edge.type)]

    @cached_property
    def parents_set(self) -> set['GraphNode']:
        """
        Return unique predecessors of current vertex
        :return: set of parents
        """
        return set(self.parents)

    @cached_property
    def children(self) -> list['GraphNode']:
        """
        Return list of successors of current vertex
        :return: list of children
        """
        return [edge.finish for edge in self.edges_from if EdgeType.is_dependency(edge.type)]

    @cached_property
    def children_set(self) -> set['GraphNode']:
        """
        Return unique successors of current vertex
        :return: set of children
        """
        return set(self.children)

    @cached_property
    def neighbors(self):
        """
        Get all edges that have types SS with current vertex
        :return: list of neighbours
        """
        return [edge.start for edge in self._parent_edges if edge.type == EdgeType.StartStart]

    @property
    def edges_to(self) -> list[GraphEdge]:
        return self._parent_edges

    @property
    def edges_from(self) -> list[GraphEdge]:
        """
        Return all successors of current vertex
        :return: list of successors
        """
        return self._children_edges

    @property
    def work_unit(self) -> WorkUnit:
        return self._work_unit

    @property
    def id(self) -> str:
        return self.work_unit.id

    @cache
    def get_inseparable_chain(self) -> Optional[list['GraphNode']]:
        """
        Gets an ordered list of whole chain of nodes, connected with edges of type INSEPARABLE_FINISH_START =
        'INSEPARABLE',
        IF self NODE IS THE START NODE OF SUCH CHAIN. Otherwise, None.

        :return: list of GraphNode or None
        """
        return [self] + self._get_inseparable_children() \
            if self.inseparable_son and not self.inseparable_parent \
            else None

    def get_inseparable_chain_with_self(self) -> list['GraphNode']:
        """
        Gets an ordered list of whole chain of nodes, connected with edges of type INSEPARABLE_FINISH_START =
        'INSEPARABLE'.

        :return: list of `inseparable chain` with starting node
        """
        return self.get_inseparable_chain() if self.get_inseparable_chain() else [self]

    def _get_inseparable_children(self) -> list['GraphNode']:
        """
        Recursively gets a child, connected with INSEPARABLE_FINISH_START edge, its inseparable child, etc.
        As any node may contain an inseparable connection with only one of its children, there is no need to choose.
        If no children are connected inseparably, returns None.

        :return: list[GraphNode]. Empty, if there is no inseparable children
        """
        inseparable_child = self.inseparable_son
        return [inseparable_child] + inseparable_child._get_inseparable_children() \
            if inseparable_child \
            else []

    def _add_child_edge(self, child: GraphEdge):
        """
        Append new edge with child

        :param child:
        :return: current graph node
        """
        self._children_edges.append(child)

    def min_start_time(self, node2swork: dict['GraphNode', ScheduledWork]) -> Time:
        return max((node2swork[edge.start].finish_time + int(edge.lag)
                    for edge in self.edges_to if edge.start in node2swork), default=Time(0))


GraphNodeDict = dict[str, GraphNode]


# TODO Make property for list of GraphEdges??
@dataclass
class WorkGraph(JSONSerializable['WorkGraph']):
    """
    Class to describe graph of works
    """
    # service vertexes
    start: GraphNode
    finish: GraphNode

    # list of works (i.e. GraphNode)
    nodes: list[GraphNode] = field(init=False)
    adj_matrix: dok_matrix = field(init=False)
    dict_nodes: GraphNodeDict = field(init=False)
    vertex_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.reinit()

    def reinit(self):
        ordered_nodes, adj_matrix, dict_nodes = self._to_adj_matrix()
        # To avoid field set of frozen instance errors
        object.__setattr__(self, 'nodes', ordered_nodes)
        object.__setattr__(self, 'adj_matrix', adj_matrix)
        object.__setattr__(self, 'dict_nodes', dict_nodes)
        object.__setattr__(self, 'vertex_count', len(ordered_nodes))

    def __hash__(self):
        return hash(self.start) + 17 * hash(self.finish)

    def __getitem__(self, item: str) -> GraphNode:
        return self.dict_nodes[item]

    def __getstate__(self):
        # custom method to avoid calling __hash__() on GraphNode objects
        representation = self._serialize()
        representation['start'] = self.start.id
        representation['finish'] = self.start.id
        return representation

    def __setstate__(self, state):
        # custom method to avoid calling __hash__() on GraphNode objects
        deserialized = self._deserialize(state)
        object.__setattr__(self, 'start', deserialized.start)
        object.__setattr__(self, 'finish', deserialized.finish)
        self.__post_init__()

    def __del__(self):
        self.dict_nodes = None
        self.start = None
        self.finish = None
        for node in self.nodes:
            node._parent_edges = None
            node._children_edges = None

    def _serialize(self) -> T:
        return {
            'nodes': [graph_node._serialize() for graph_node in self.nodes]
        }

    @classmethod
    def _deserialize(cls, representation: T) -> JS:
        serialized_nodes = [GraphNode._deserialize(node) for node in representation['nodes']]
        assert not serialized_nodes[0]['parent_edges']
        start_id, finish_id = (serialized_nodes[i]['work_unit'].id for i in (0, -1))

        nodes_dict = dict()
        for node_info in serialized_nodes:
            wu, parent_info = (node_info[member] for member in ('work_unit', 'parent_edges'))
            graph_node = GraphNode(wu, [(nodes_dict[p_id], p_lag, p_type) for p_id, p_lag, p_type in parent_info])
            nodes_dict[wu.id] = graph_node

        return WorkGraph(nodes_dict[start_id], nodes_dict[finish_id])

    # TODO: Check that adj matrix is really need
    def _to_adj_matrix(self) -> tuple[list[GraphNode], dok_matrix, dict[str, GraphNode]]:
        """
        Build adjacency matrix from current graph
        """
        ordered_nodes: list[GraphNode] = list(self.start.traverse_children(topologically=True))
        node2ind: dict[GraphNode, int] = {
            v: i for i, v in enumerate(ordered_nodes)
        }
        id2node: dict[str, GraphNode] = {node.id: node for node in node2ind.keys()}
        adj_mx = dok_matrix((len(node2ind), len(node2ind)), dtype=np.short)
        weight = 0
        for v, i in node2ind.items():
            for child in v.children:
                c_i = node2ind[child]
                weight = max((w_req.volume for w_req in v.work_unit.worker_reqs), default=0.000001)
                adj_mx[i, c_i] = weight

        return ordered_nodes, adj_mx, id2node
