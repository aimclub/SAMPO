from abc import ABC, abstractmethod
from random import Random
from typing import Iterator

from sampo.schemas import WorkGraph, GraphNode, WorkTimeEstimator
from sampo.schemas.graph import get_start_stage
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
        return WorkGraph.from_nodes(list(self.iterate()), self._rand)

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
                 work_estimator: WorkTimeEstimator,
                 start: GraphNode,
                 node2followers: dict[str, list[tuple[list[GraphNode], float]]],
                 averages: dict[str, float]):
        super().__init__(rand)
        self._start = start
        self._averages = averages
        self._node2followers = node2followers
        self._work_estimator = work_estimator

    def first(self):
        return self._start

    def iterate(self) -> Iterator[GraphNode]:
        queue = copy_nodes([self._start], drop_outer_works=True)

        # should make things in topological way..

        seen = set()
        seen_queue = set()
        # seen.add(self._start)

        while queue:
            node = queue.pop()

            assert node not in seen, 'Duplicate returned'

            additions = self.next(node)

            yield node
            seen.add(node)

            # TODO now performing like DFS, swap places to make BFS
            new_queue = [task for subgraph in additions for task in subgraph if task not in seen_queue]

            seen_queue.update(new_queue)


            # assert len(new_queue) == len(set(new_queue)), 'Duplicate in generation!'
            # assert len(queue) == len(set(queue)), 'Duplicate in queue!'
            # new_queue_1 = new_queue + queue

            # assert len(new_queue_1) == len(set(new_queue_1)), 'Duplicate in new queue!'

            # assert len(queue) == len(set(queue)), 'Duplicate in queue!'

            new_queue.extend(queue)
            queue = new_queue

            # assert len(new_queue) == len(set(new_queue)), 'Duplicate in new queue!'

            # print(len(queue))

    def next(self, node: GraphNode, min_prob: float = 0, max_prob: float = 1) -> list[list[GraphNode]] | None:
        result = self._node2followers.get(node.id, None)
        if result is None:
            return []
        generated_subgraphs = [(prob, copy_nodes(nodes, drop_outer_works=True)) for nodes, prob in result if prob >= min_prob and prob <= max_prob and self._rand.random() < prob]
        inner_start = get_start_stage(rand=self._rand)
        inner_start.add_parents([node])

        for prob, subgraph in generated_subgraphs:
            if prob < 1:
                node.add_followers(subgraph, 0)

        generated_subgraphs = [v for _, v in generated_subgraphs]

        for subgraph in generated_subgraphs:
            add_default_predecessor(subgraph, inner_start)

        node.add_followers([inner_start], 0)
        generated_subgraphs.append([inner_start])
        return generated_subgraphs

    def average_labor_cost(self, node: GraphNode):
        """
        Returns the labor cost for the given node plus average following subgraph
        """
        return self._averages.get(node.id, 0)


class ProbabilisticFollowingStochasticGraphScheme(StochasticGraphScheme):
    def __init__(self,
                 rand: Random,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 wg: WorkGraph = None):
        super().__init__(rand)
        self._fixed_graph = wg.copy()
        self._node2followers = {node.id: [([child], 1.0) for child in node.children]
                                for node in self._fixed_graph.nodes}
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
        # # add followers
        # for follower_info in self._node2followers.values():
        #     for follower_subgraph, _ in follower_info:
        #         for follower in follower_subgraph:
        #             if follower.id not in averages:
        #                 averages[follower.id] = self._work_priority_f(follower, self._working_time_f, self._work_estimator)

        return ProbabilisticFollowingStochasticGraph(self._rand, self._work_estimator, self._fixed_graph.start,
                                                     self._node2followers, averages)
