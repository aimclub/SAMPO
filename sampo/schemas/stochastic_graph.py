from abc import ABC, abstractmethod
from random import Random
from typing import Iterable

from sampo.schemas import WorkGraph, GraphNode


class StochasticGraphSchemeInterface(ABC):

    def __init__(self, rand: Random):
        self._rand = rand

    @abstractmethod
    def to_work_graph(self) -> WorkGraph:
        """
        Fully simulates stochastic process and returns sample
        work graph satisfies the stochastic scheme
        """
        ...

    @abstractmethod
    def iterate(self) -> Iterable[GraphNode]:
        """
        Returns the iterable of the resulting stochastic graph,
        which is generated on the fly. This is true stochastic process
        """
        ...

    @abstractmethod
    def generate_next(self, node: GraphNode) -> WorkGraph | None:
        """
        Returns generated WorkGraph following given node
        or None if nothing was generated
        """
        ...

    @abstractmethod
    def next(self, node: GraphNode) -> GraphNode | None:
        """
        Returns next node in the resulting graph.
        This can be generated or initial node, you don't know it.
        """
        ...

