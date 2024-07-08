from abc import ABC, abstractmethod
from random import Random
from typing import Iterable, Iterator

from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade, work_priority
from sampo.schemas import WorkGraph, GraphNode, WorkTimeEstimator
from sampo.schemas.time_estimator import DefaultWorkEstimator


class StochasticGraph(ABC):

    def __init__(self, rand: Random):
        self._rand = rand

    def to_work_graph(self) -> WorkGraph:
        """
        Fully simulates stochastic process and returns sample
        work graph satisfies the stochastic scheme
        """
        return WorkGraph.from_nodes(list(self.iterate()))

    @abstractmethod
    def first(self):
        ...

    @abstractmethod
    def iterate(self) -> Iterator[GraphNode]:
        """
        Returns the iterable of the resulting stochastic graph,
        which is generated on the fly. This is true stochastic process
        """
        ...

    @abstractmethod
    def generate_next(self, node: GraphNode) -> list[list[GraphNode]] | None:
        """
        Returns generated WorkGraph following given node
        or None if nothing was generated
        """
        ...

    @abstractmethod
    def next(self, node: GraphNode, min_prob: float = 0) -> list[list[GraphNode]] | None:
        """
        Returns next node in the resulting graph.
        This can be generated or initial node, you don't know it.
        """
        ...

    @abstractmethod
    def average_labor_cost(self, node: GraphNode):
        """
        Returns the labor cost for the given node plus average following subgraph
        """
        ...


class StochasticGraphScheme(ABC):

    def __init__(self, rand: Random):
        self._rand = rand

    @abstractmethod
    def prepare_graph(self) -> StochasticGraph:
        """
        Fully simulates stochastic process and returns sample
        work graph satisfies the stochastic scheme
        """
        ...


class ProbabilisticFollowingStochasticGraph(StochasticGraph):

    def __init__(self,
                 rand: Random,
                 start: GraphNode,
                 node2followers: dict[str, list[tuple[list[GraphNode], float]]],
                 averages: dict[str, float]):
        super().__init__(rand)
        self._node2followers = node2followers
        self._start = start
        self._averages = averages

    def first(self):
        return self._start

    def iterate(self) -> Iterator[GraphNode]:
        node = self._start
        yield node
        while (node := self.next(node)) is not None:
            yield node

    def generate_next(self, node: GraphNode) -> list[list[GraphNode]] | None:
        return self.next(node, 1)

    def next(self, node: GraphNode, min_prob: float = 0) -> list[list[GraphNode]] | None:
        result = self._node2followers.get(node.work_unit.name, None)
        if result is None:
            return None
        return [nodes for nodes, prob in result if self._rand.random() < prob]

    def average_labor_cost(self, node: GraphNode):
        """
        Returns the labor cost for the given node plus average following subgraph
        """
        return self._averages[node.work_unit.name]


class ProbabilisticFollowingStochasticGraphScheme(StochasticGraphScheme):
    def __init__(self,
                 rand: Random,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 wg: WorkGraph = None):
        super().__init__(rand)
        self._fixed_graph = wg
        self._node2followers = {node.id: [([child], 1.0) for child in edge.children]
                                for node in wg.nodes for edge in node.children}
        self._work_estimator = work_estimator

    def add_part(self, node: str, nodes: list[GraphNode], prob: float):
        followers = self._node2followers.get(node, None) or []
        followers.append((nodes, prob))
        self._node2followers[node] = followers

    def _get_subgraph_labor(self, entry: tuple[list[GraphNode], float]):
        nodes, prob = entry
        return prob * sum(work_priority(node, calculate_working_time_cascade, self._work_estimator)
                          for node in nodes)

    def prepare_graph(self) -> StochasticGraph:
        averages = {node.id: sum(self._get_subgraph_labor(entry)
                                 for entry in self._node2followers[node.work_unit.name])
                    for node in self._fixed_graph.nodes}
        return ProbabilisticFollowingStochasticGraph(self._rand, self._fixed_graph.start,
                                                     self._node2followers, averages)
