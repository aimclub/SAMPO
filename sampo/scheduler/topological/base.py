from typing import Optional

import numpy as np
from toposort import toposort_flatten, toposort

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


def get_node_dependencies(wg: WorkGraph) -> dict[str, set[str]]:
    """
    Creates a mapper for nodes in Word Graph that matches each node id to its parents ids
    and leaves only the first node in inseparable chains.

    :param wg: WorkGraph

    :return: dict that maps node id with set of parent nodes ids
    """
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    node_id2inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            node_id2inseparable_parents[child.id] = node.id

    # here we aggregate information about relationships from the whole inseparable chain
    node_id2children = {node.id: set([node_id2inseparable_parents[child.id]
                                      for inseparable in node.get_inseparable_chain_with_self()
                                      for child in inseparable.children]) - {node.id}
                        for node in nodes}

    node_id2parents = {node.id: set() for node in nodes}
    for node_id, children_nodes in node_id2children.items():
        for child_id in children_nodes:
            node_id2parents[child_id].add(node_id)

    return node_id2parents


class TopologicalScheduler(GenericScheduler):
    """
    Scheduler, that represent 'WorkGraph' in topological order.
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.Topological,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=AverageReqResourceOptimizer(),
                         timeline_type=MomentumTimeline,
                         prioritization_f=self._topological_sort,
                         optimize_resources_f=self.get_default_res_opt_function(lambda _: Time(0)),
                         work_estimator=work_estimator)

    # noinspection PyMethodMayBeStatic
    def _topological_sort(self, wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
        """
        Sort 'WorkGraph' in topological order.

        :param wg: WorkGraph
        :param work_estimator: function that calculates execution time of the work
        :return: list of sorted nodes in graph
        """
        dependencies = get_node_dependencies(wg)

        tsorted_nodes_ids = toposort_flatten(dependencies, sort=True)
        tsorted_nodes = [wg[node_id] for node_id in tsorted_nodes_ids]

        return list(reversed(tsorted_nodes))


class RandomizedTopologicalScheduler(TopologicalScheduler):
    """
    Scheduler, that represent 'WorkGraph' in topological order with random.
    """

    def __init__(self,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator=work_estimator)
        self._random_state = np.random.RandomState(random_seed)

    def _topological_sort(self, wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
        def shuffle(nodes: set[str]) -> list[str]:
            """
            Shuffle nodes that are on the same level.

            :param nodes: list of nodes
            :return: list of shuffled indices
            """
            nds = list(nodes)
            indices = np.arange(len(nds))
            self._random_state.shuffle(indices)
            return [nds[ind] for ind in indices]

        dependencies = get_node_dependencies(wg)
        tsorted_nodes: list[GraphNode] = [
            wg[node_id] for level in toposort(dependencies)
            for node_id in shuffle(level)
        ]

        return list(reversed(tsorted_nodes))
