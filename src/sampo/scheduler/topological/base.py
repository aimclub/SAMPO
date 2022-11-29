from typing import List, Dict, Set, Optional, Iterable

import numpy as np
from toposort import toposort_flatten, toposort

from sampo.scheduler.base import Scheduler
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.utils.momentum_timeline import create_timeline, schedule_with_time_spec
from sampo.scheduler.utils.multi_contractor import get_best_contractor_and_worker_borders
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.validation import validate_schedule


class TopologicalScheduler(Scheduler):

    def __init__(self, scheduler_type: SchedulerType = SchedulerType.Topological,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=AverageReqResourceOptimizer(),
                         work_estimator=work_estimator)

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 validate: bool = False) \
            -> Schedule:
        # Checking pre-conditions for this scheduler_topological to be applied
        # check_all_workers_have_same_qualification(wg, contractors)

        tsorted_nodes: List[GraphNode] = self._topological_sort(wg)

        schedule = Schedule.from_scheduled_works(
            self.build_scheduler(tsorted_nodes, contractors, spec, self.work_estimator), wg
        )

        # check the validity received scheduler
        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule

    # noinspection PyMethodMayBeStatic
    def _topological_sort(self, wg: WorkGraph) -> List[GraphNode]:
        node2ind = {node: i for i, node in enumerate(wg.nodes)}

        dependents: Dict[int, Set[int]] = {
            node2ind[v]: {node2ind[node] for node in v.parents}
            for v in wg.nodes
        }

        tsorted_nodes_indices: List[int] = toposort_flatten(dependents, sort=True)
        tsorted_nodes = [wg.nodes[i] for i in tsorted_nodes_indices]
        return tsorted_nodes

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, tasks: List[GraphNode], contractors: List[Contractor],
                        spec: ScheduleSpec,
                        work_estimator: WorkTimeEstimator = None) \
            -> Iterable[ScheduledWork]:
        """
        Builds a schedule from a list of tasks where all dependents tasks are guaranteed to be later
        in the sequence than their dependencies
        :param work_estimator:
        :param tasks: list of tasks ordered by some algorithm according to their dependencies and priorities
        :param spec: spec for current scheduling
        :param contractors: pools of workers available for execution
        :return: a schedule
        """

        # data structure to hold scheduled tasks
        node2swork: Dict[GraphNode, ScheduledWork] = dict()

        timeline = create_timeline(tasks, contractors)

        # we can get agents here, because they are always same and not updated
        worker_pool = get_worker_contractor_pool(contractors)

        # We allocate resources for the whole inseparable chain, when we process the first node in it.
        # So, we will store IDs of non-head nodes in such chains to skip them.
        # Note that tasks are already topologically ordered,
        # i.e., the first node in a chain is always processed before its children

        skipped_inseparable_children: Set[str] = set()
        # scheduling all the tasks in a one-by-one manner
        for index, node in enumerate(tasks):
            # skip, if this node was processed as a part of an inseparable chin previously
            if node.id in skipped_inseparable_children:
                continue
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)

            # 0. find, if the node starts an inseparable chain

            inseparable_chain = node.get_inseparable_chain()
            if inseparable_chain:
                skipped_inseparable_children.update((ch.id for ch in inseparable_chain))
            whole_work_nodes = inseparable_chain if inseparable_chain else [node]

            min_count_worker_team, max_count_worker_team, contractor, workers = \
                get_best_contractor_and_worker_borders(worker_pool, contractors, node.work_unit.worker_reqs)

            best_worker_team = [worker.copy() for worker in workers]

            # apply worker team spec
            self.optimize_resources_using_spec(work_unit, best_worker_team, work_spec,
                                               lambda optimize_array: self.resource_optimizer.optimize_resources(
                                                   worker_pool, best_worker_team,
                                                   optimize_array,
                                                   min_count_worker_team, max_count_worker_team,
                                                   # dummy
                                                   lambda _: Time(0)))

            # finish scheduling with time spec
            schedule_with_time_spec(index, node, node2swork, whole_work_nodes, timeline, best_worker_team, contractor,
                                    work_spec.assigned_time, work_estimator)

        return node2swork.values()


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

        dependents: Dict[GraphNode, Set[GraphNode]] = {v: v.parents_set for v in wg.nodes}
        tsorted_nodes: List[GraphNode] = [
            node for level in toposort(dependents)
            for node in shuffle(level)
        ]

        # print([node.work_unit.id for node in tsorted_nodes])
        return tsorted_nodes
