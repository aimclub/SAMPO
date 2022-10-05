from typing import List, Dict, Set, Optional, Tuple, Any
from toposort import toposort_flatten, toposort

import numpy as np

from schemas.work_estimator import WorkTimeEstimator
from scheduler.topological.schedule_validation import validate_schedule as validate_topological_schedule
from scheduler.base import SchedulerType
from schemas.schedule import Schedule
from scheduler.base import Scheduler
from scheduler.topological.schedule_builder import build_schedule_from_task_sequence
from schemas.contractor import Contractor
from schemas.graph import GraphNode, WorkGraph


class TopologicalScheduler(Scheduler):
    scheduler_type: SchedulerType = SchedulerType.Topological

    def __init__(self, work_estimator: Optional[WorkTimeEstimator or None] = None):
        self.work_estimator = work_estimator

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 ksg_info: Dict[str, Dict[str, Any]],
                 start: str,
                 validate_schedule: Optional[bool] = False) \
            -> Tuple[Schedule, List[str]]:
        # Checking pre-conditions for this scheduler_topological to be applied
        # check_all_workers_have_same_qualification(wg, contractors)

        tsorted_nodes: List[GraphNode] = self._topological_sort(wg)

        scheduled_works = build_schedule_from_task_sequence(tsorted_nodes, wg, contractors, self.work_estimator)
        schedule = Schedule.from_scheduled_works(scheduled_works, ksg_info, start)

        # check the validity received scheduler
        if validate_schedule:
            validate_topological_schedule(schedule, wg, contractors)

        ordered_work_id = [swork.work_unit.id for swork in sorted(scheduled_works, key=lambda x: x.start_end_time[0])]

        return schedule, ordered_work_id

    # noinspection PyMethodMayBeStatic
    def _topological_sort(self, wg: WorkGraph) -> List[GraphNode]:
        node2ind = {node: i for i, node in enumerate(wg.nodes)}

        dependents: Dict[int, Set[int]] = {
            node2ind[v]: {node2ind[node] for node in v.parent_nodes}
            for v in wg.nodes
        }

        tsorted_nodes_indices: List[int] = toposort_flatten(dependents, sort=True)
        tsorted_nodes = [wg.nodes[i] for i in tsorted_nodes_indices]
        return tsorted_nodes


class RandomizedTopologicalScheduler(TopologicalScheduler):
    def __init__(self, work_estimator: Optional[WorkTimeEstimator or None] = None,
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator)
        self._random_state = np.random.RandomState(random_seed)

    def _topological_sort(self, wg: WorkGraph) -> List[GraphNode]:
        def shuffle(nodes: Set[GraphNode]) -> List[GraphNode]:
            nds = list(nodes)
            indices = np.arange(len(nds))
            self._random_state.shuffle(indices)
            return [nds[ind] for ind in indices]

        dependents: Dict[GraphNode, Set[GraphNode]] = {v: set(v.parent_nodes) for v in wg.nodes}
        tsorted_nodes: List[GraphNode] = [
            node for level in toposort(dependents)
            for node in shuffle(level)
        ]

        # print([node.work_unit.id for node in tsorted_nodes])
        return tsorted_nodes
