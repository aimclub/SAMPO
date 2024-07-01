from abc import ABC, abstractmethod
from random import Random
from typing import Iterable

from sampo.schemas import WorkGraph, GraphNode


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
    def iterate(self) -> Iterable[GraphNode]:
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
                 node2followers: dict[str, list[tuple[list[GraphNode], float]]]):
        super().__init__(rand)
        self._node2followers = node2followers
        self._start = start

    def iterate(self) -> Iterable[GraphNode]:
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


class ProbabilisticFollowingStochasticGraphScheme(StochasticGraphScheme):
    def __init__(self, rand: Random, wg: WorkGraph = None):
        super().__init__(rand)
        self._fixed_graph = wg
        self._node2followers = {node.id: [([child], 1) for child in edge.children]
                                for node in wg.nodes for edge in node.children}

    def add_part(self, node: str, nodes: list[GraphNode], prob: float):
        followers = self._node2followers.get(node, None) or []
        followers.append((nodes, prob))
        self._node2followers[node] = followers

    def prepare_graph(self) -> StochasticGraph:
        return ProbabilisticFollowingStochasticGraph(self._rand, self._fixed_graph.start, self._node2followers)
