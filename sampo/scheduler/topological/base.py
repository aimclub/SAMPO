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
        node2ind = {node: i for i, node in enumerate(wg.nodes)}

        dependents: dict[int, set[int]] = {
            node2ind[v]: {node2ind[node] for node in v.parents}
            for v in wg.nodes
        }

        tsorted_nodes_indices: list[int] = toposort_flatten(dependents, sort=True)
        tsorted_nodes = [wg.nodes[i] for i in tsorted_nodes_indices]
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
        def shuffle(nodes: set[GraphNode]) -> list[GraphNode]:
            """
            Shuffle nodes that are on the same level.

            :param nodes: list of nodes
            :return: list of shuffled indices
            """
            nds = list(nodes)
            indices = np.arange(len(nds))
            self._random_state.shuffle(indices)
            return [nds[ind] for ind in indices]

        dependents: dict[GraphNode, set[GraphNode]] = {v: v.parents_set for v in wg.nodes}
        tsorted_nodes: list[GraphNode] = [
            node for level in toposort(dependents)
            for node in shuffle(level)
        ]

        # print([node.work_unit.id for node in tsorted_nodes])
        return list(reversed(tsorted_nodes))
