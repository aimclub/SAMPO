from typing import Optional

import numpy as np
from toposort import toposort

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.schemas.graph import GraphNode
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
    def _topological_sort(self, head_nodes: list[GraphNode],
                          node_id2parent_ids: dict[str, set[str]],
                          node_id2child_ids: dict[str, set[str]],
                          work_estimator: WorkTimeEstimator) -> list[GraphNode]:
        """
        Sort 'WorkGraph' in topological order.

        :param head_nodes: A list sorted in topological order and containing only the head nodes
        :param work_estimator: function that calculates execution time of the work
        :return: list of sorted nodes in graph
        """
        return list(reversed(head_nodes))


class RandomizedTopologicalScheduler(TopologicalScheduler):
    """
    Scheduler, that represent 'WorkGraph' in topological order with random.
    """

    def __init__(self,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator=work_estimator)
        self._random_state = np.random.RandomState(random_seed)

    def _topological_sort(self, head_nodes: list[GraphNode],
                          node_id2parent_ids: dict[str, set[str]],
                          node_id2child_ids: dict[str, set[str]],
                          work_estimator: WorkTimeEstimator) -> list[GraphNode]:
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

        tsorted_node_ids: list[str] = [
            node_id for level in toposort(node_id2parent_ids)
            for node_id in shuffle(level)
        ]

        ordered_nodes = sorted(head_nodes, key=lambda node: tsorted_node_ids.index(node.id), reverse=True)

        return ordered_nodes
