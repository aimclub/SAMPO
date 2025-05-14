from abc import ABC, abstractmethod
from random import Random
from typing import Iterator

from sampo.schemas import WorkGraph, GraphNode, WorkTimeEstimator
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.nodes import copy_nodes, add_default_predecessor


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
        self._start = start
        self._averages = averages
        self._node2followers = node2followers

    def first(self):
        return self._start

    def iterate(self) -> Iterator[GraphNode]:
        queue = copy_nodes([self._start], drop_outer_works=True)

        while queue:
            node = queue.pop()
            yield node

            additions = self.next(node)
            # TODO now performing like DFS, swap places to make BFS
            new_queue = [task for subgraph in additions for task in subgraph]
            new_queue.extend(queue)
            queue = new_queue

    def next(self, node: GraphNode, min_prob: float = 0) -> list[list[GraphNode]] | None:
        result = self._node2followers.get(node.id, None)
        if result is None:
            return []
        generated_subgraph = [copy_nodes(nodes, drop_outer_works=True) for nodes, prob in result if prob >= min_prob and self._rand.random() < prob]
        for subgraph in generated_subgraph:
            add_default_predecessor(subgraph, node)
        return generated_subgraph

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
        self._fixed_graph = wg.copy()
        self._node2followers = {node.id: [([child], 1.0) for child in edge.children]
                                for node in wg.nodes for edge in node.children}
        self._work_estimator = work_estimator

        from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade, work_priority

        self._working_time_f = calculate_working_time_cascade
        self._work_priority_f = work_priority

    def add_part(self, node: str, nodes: list[GraphNode], prob: float):
        followers = self._node2followers.get(node, [])
        followers.append((nodes, prob))
        self._node2followers[node] = followers

    def _get_subgraph_labor(self, entry: tuple[list[GraphNode], float]) -> float:
        nodes, prob = entry
        if not nodes:
            return 0
        return prob * sum(self._work_priority_f(node, self._working_time_f, self._work_estimator)
                          for node in nodes)

    def prepare_graph(self) -> StochasticGraph:
        averages = {node.id: sum(self._get_subgraph_labor(entry)
                                 for entry in self._node2followers.get(node.id, []))
                    for node in self._fixed_graph.nodes}
        return ProbabilisticFollowingStochasticGraph(self._rand, self._fixed_graph.start,
                                                     self._node2followers, averages)
