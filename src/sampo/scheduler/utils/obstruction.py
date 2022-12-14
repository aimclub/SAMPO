from abc import ABC, abstractmethod
from copy import deepcopy
from random import Random
from typing import Callable

from sampo.schemas.graph import WorkGraph, GraphNode


class Obstruction(ABC):
    """
    Tests the probability and, if it's true, apply the obstruction.
    """
    def __init__(self, probability: float, rand: Random):
        self._probability = probability
        self._rand = rand

    def generate(self, wg: WorkGraph):
        """
        Tests the probability and, if it's true, apply the obstruction.

        :param wg: given WorkGraph
        """
        if self._rand.random() < self._probability:
            self.apply(wg)

    @abstractmethod
    def apply(self, wg: WorkGraph):
        """
        The main method of the obstruction.

        Should apply provided obstruction to the given WorkGraph in-place.

        :param wg: given WorkGraph
        """
        ...


class OneInsertObstruction(Obstruction):
    """
    Applying seeks the random part of given WorkGraph and inserts it into that point
    """

    def __init__(self, probability: float, rand: Random, insert_wg_getter: Callable[[Random], WorkGraph]):
        super().__init__(probability, rand)
        self._insert_wg_getter = insert_wg_getter

    @staticmethod
    def from_static_graph(probability: float, rand: Random, insert_wg: WorkGraph) -> 'OneInsertObstruction':
        return OneInsertObstruction(probability, rand, lambda _: deepcopy(insert_wg))

    def apply(self, wg: WorkGraph):
        # get the insert graph
        insert_wg = self._insert_wg_getter(self._rand)
        # get the insert point
        insert_node: GraphNode = self._rand.sample(wg.nodes, 1)[0]
        # insert
        insert_wg.start.add_parents([insert_node])

        if insert_node.children:
            # get the insert end point
            insert_node_child: GraphNode = self._rand.sample(insert_node.children, 1)[0]
            # insert
            insert_node_child.add_parents([insert_wg.start])

