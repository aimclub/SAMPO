from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property, cache
from typing import List, Union, Tuple, Optional, Dict, Set

import numpy as np
from scipy.sparse import dok_matrix

from sampo.schemas.serializable import JSONSerializable, T, JS
from sampo.schemas.works import WorkUnit


# TODO: describe the class (description, parameters)
class EdgeType(Enum):
    InseparableFinishStart = 'IFS'
    LagFinishStart = 'FFS'
    StartStart = 'SS'
    FinishFinish = 'FF'
    FinishStart = 'FS'
    StartFinish = 'SF'

# TODO: describe the class (description, parameters)
@dataclass
class GraphEdge:
    start: 'GraphNode'
    finish: 'GraphNode'
    lag: Optional[float] = 0
    type: Optional[EdgeType] = None


# TODO: describe the class (description, parameters)
class GraphNode(JSONSerializable['GraphNode']):
    _work_unit: WorkUnit
    _parent_edges: List[GraphEdge]
    _children_edges: List[GraphEdge]

    # TODO: describe the function (description, parameters, return type)
    def __init__(self, work_unit: WorkUnit,
                 parent_works: Union[List['GraphNode'], List[Tuple['GraphNode', float, EdgeType]]]):
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

    # TODO: describe the function (description, return type)
    def _serialize(self) -> T:
        return {
            'work_unit': self._work_unit._serialize(),
            'parent_edges': [(e.start.id, e.lag, e.type.value) for e in self._parent_edges],
            # 'child_edges': [(e.finish.work_unit.id, e.lag, e.type.value) for e in self._children_edges]
        }

    # TODO: describe the function (description, parameters, return type)
    @classmethod
    def _deserialize(cls, representation: T) -> dict:
        representation['work_unit'] = WorkUnit._deserialize(representation['work_unit'])
        representation['parent_edges'] = [(e[0], e[1], EdgeType(e[2])) for e in representation['parent_edges']]
        # representation['child_edges'] = [(e[0], e[1], EdgeType(e[2])) for e in representation['child_edges']]
        return representation

    # TODO: describe the function (description, parameters, return type)
    def update_work_unit(self, work_unit: WorkUnit) -> None:
        self._work_unit = work_unit

    # TODO: add a comment about the returned type and complete the description
    def add_parents(self, parent_works: List['GraphNode'] or List[Tuple['GraphNode', float, EdgeType]]) -> None:
        """
        Two-sided linking of children and parents

        :param parent_works:
        """
        edges: List[GraphEdge] = []
        if len(parent_works) > 0 and isinstance(parent_works[0], GraphNode):
            edges = [GraphEdge(p, self, -1, EdgeType.FinishStart) for p in parent_works]
        elif len(parent_works) > 0 and isinstance(parent_works[0], tuple):
            edges = [GraphEdge(p, self, lag, edge_type) for p, lag, edge_type in parent_works]

        for i, p in enumerate(parent_works):
            p: GraphNode = p[0] if isinstance(p, tuple) else p
            p._add_child_edge(edges[i])
        self._parent_edges += edges

    # TODO: describe the function (description, return type)
    def is_inseparable_parent(self) -> bool:
        return self.inseparable_son is not None

    # TODO: describe the function (description, return type)
    def is_inseparable_son(self) -> bool:
        return self.inseparable_parent is not None

    # TODO: describe the function (description, parameters, return type)
    def traverse_children(self, topologically: bool = False):
        visited_vertexes = set()
        vertexes_to_visit = deque([self])
        while len(vertexes_to_visit) > 0:
            v = vertexes_to_visit.popleft()
            if topologically and any(p not in visited_vertexes for p in v.parents):
                vertexes_to_visit.append(v)
                continue
            if v not in visited_vertexes:
                visited_vertexes.add(v)
                vertexes_to_visit.extend(v.children)
                yield v

    # TODO: describe the function (description, return type)
    @cached_property
    def inseparable_son(self) -> Optional['GraphNode']:
        inseparable_children = list([x.finish for x in self._children_edges
                                     if x.type == EdgeType.InseparableFinishStart])
        return inseparable_children[0] if inseparable_children else None

    # TODO: describe the function (description, return type)
    @cached_property
    def inseparable_parent(self) -> Optional['GraphNode']:
        inseparable_parents = list([x.start for x in self._parent_edges if x.type == EdgeType.InseparableFinishStart])
        return inseparable_parents[0] if inseparable_parents else None

    # TODO Describe the function (description, return type)
    @cached_property
    def parents(self) -> List['GraphNode']:
        return list([edge.start for edge in self._parent_edges])

    # TODO Describe the function (description, return type)
    @cached_property
    def parents_set(self) -> Set['GraphNode']:
        return set(self.parents)

    # TODO Describe the function (description, return type)
    @cached_property
    def children(self) -> List['GraphNode']:
        return list([edge.finish for edge in self._children_edges])

    # TODO Describe the function (description, return type)
    @cached_property
    def children_set(self) -> Set['GraphNode']:
        return set(self.children)

    @property
    def edges_to(self) -> List[GraphEdge]:
        return self._parent_edges

    # TODO: describe the function (description, return type)
    @property
    def edges_from(self) -> List[GraphEdge]:
        return self._children_edges

    # TODO: describe the function (description, return type)
    @property
    def work_unit(self) -> WorkUnit:
        return self._work_unit

    # TODO: describe the function (description, return type)
    @property
    def id(self) -> str:
        return self.work_unit.id

    @cache
    def get_inseparable_chain(self) -> Optional[List['GraphNode']]:
        """
        Gets an ordered list of whole chain of nodes, connected with edges of type INSEPARABLE_FINISH_START =
        'INSEPARABLE',
        IF self NODE IS THE START NODE OF SUCH CHAIN. Otherwise, None.

        :return: List of GraphNode or None
        """
        return [self] + self._get_inseparable_children() \
            if bool(self.inseparable_son) and not bool(self.inseparable_parent) \
            else None

    def get_inseparable_chain_with_self(self) -> List['GraphNode']:
        """
        Gets an ordered list of whole chain of nodes, connected with edges of type INSEPARABLE_FINISH_START =
        'INSEPARABLE'.

        :return: List of `inseparable chain` with starting node
        """
        return self.get_inseparable_chain() if self.get_inseparable_chain() else [self]

    def _get_inseparable_children(self) -> List['GraphNode']:
        """
        Recursively gets a child, connected with INSEPARABLE_FINISH_START edge, its inseparable child, etc.
        As any node may contain an inseparable connection with only one of its children, there is no need to choose.
        If no children are connected inseparably, returns None.

        :return: List[GraphNode]. Empty, if there is no inseparable children
        """
        inseparable_child = self.inseparable_son
        return [inseparable_child] + inseparable_child._get_inseparable_children() \
            if inseparable_child \
            else []

    # TODO: describe the function (description, parameters, return type)
    def _add_child_edge(self, child: GraphEdge):
        self._children_edges.append(child)


GraphNodeDict = Dict[str, GraphNode]


# TODO: describe the class (description, parameters)
# TODO Make property for list of GraphEdges??
@dataclass(frozen=True)
class WorkGraph(JSONSerializable['WorkGraph']):
    start: GraphNode
    finish: GraphNode

    nodes: List[GraphNode] = field(init=False)
    adj_matrix: dok_matrix = field(init=False)
    dict_nodes: GraphNodeDict = field(init=False)
    vertex_count: int = field(init=False)

    # TODO: describe the function (description, parameters, return type)
    def __post_init__(self) -> None:
        ordered_nodes, adj_matrix, dict_nodes = self._to_adj_matrix()
        # To avoid field set of frozen instance errors
        object.__setattr__(self, 'nodes', ordered_nodes)
        object.__setattr__(self, 'adj_matrix', adj_matrix)
        object.__setattr__(self, 'dict_nodes', dict_nodes)
        object.__setattr__(self, 'vertex_count', len(ordered_nodes))

    def __hash__(self):
        return hash(self.start) + hash(self.finish)

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

    # TODO: describe the function (description, return type)
    def _serialize(self) -> T:
        return {
            'nodes': [graph_node._serialize() for graph_node in self.nodes]
        }

    # TODO: describe the function (description, parameters, return type)
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

    # TODO: describe the function (description, return type)
    # TODO Check that adj matrix is really need
    def _to_adj_matrix(self) -> Tuple[List[GraphNode], dok_matrix, Dict[str, GraphNode]]:
        ordered_nodes: List[GraphNode] = list(self.start.traverse_children(topologically=True))
        node2ind: Dict[GraphNode, int] = {
            v: i for i, v in enumerate(ordered_nodes)
        }
        id2node: Dict[str, GraphNode] = {node.id: node for node in node2ind.keys()}
        adj_mx = dok_matrix((len(node2ind), len(node2ind)), dtype=np.short)
        for v, i in node2ind.items():
            for child in v.children:
                c_i = node2ind[child]
                try:
                    weight = max((w_req.volume for w_req in v.work_unit.worker_reqs), default=0.000001)
                except:
                    print('ln')
                adj_mx[i, c_i] = weight

        return ordered_nodes, adj_mx, id2node
