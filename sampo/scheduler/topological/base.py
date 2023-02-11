from typing import List, Dict, Set, Optional, Iterable

import numpy as np
from toposort import toposort_flatten, toposort

from sampo.scheduler.base import Scheduler
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.scheduler.utils.multi_contractor import get_worker_borders, run_contractor_search
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.resources import Worker
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

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline]:
        tsorted_nodes: List[GraphNode] = self._topological_sort(wg)

        schedule, schedule_start_time, timeline = \
            self.build_scheduler(tsorted_nodes, contractors, spec, self.work_estimator, assigned_parent_time, timeline)
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline

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
                        work_estimator: WorkTimeEstimator = None,
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, MomentumTimeline]:
        """
        Builds a schedule from a list of tasks where all dependents tasks are guaranteed to be later
        in the sequence than their dependencies

        :param tasks: list of tasks ordered by some algorithm according to their dependencies and priorities
        :param spec: spec for current scheduling
        :param contractors: pools of workers available for execution
        :param timeline: the previous used timeline can be specified to handle previously scheduled works
        :param assigned_parent_time: start time of the whole schedule(time shift)
        :param work_estimator:
        :return: the schedule
        """

        # data structure to hold scheduled tasks
        node2swork: Dict[GraphNode, ScheduledWork] = dict()

        # we can get agents here, because they are always same and not updated
        worker_pool = get_worker_contractor_pool(contractors)

        if not isinstance(timeline, MomentumTimeline):
            timeline = MomentumTimeline(tasks, contractors, worker_pool)

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

            def run_with_contractor(contractor: Contractor) -> tuple[Time, Time, List[Worker]]:
                min_count_worker_team, max_count_worker_team, workers = \
                    get_worker_borders(worker_pool, contractor, node.work_unit.worker_reqs)

                if len(workers) != len(work_unit.worker_reqs):
                    return Time(0), Time.inf(), []

                worker_team = [worker.copy() for worker in workers]

                # apply worker team spec
                self.optimize_resources_using_spec(work_unit, worker_team, work_spec,
                                                   lambda optimize_array: self.resource_optimizer.optimize_resources(
                                                       worker_pool, worker_team,
                                                       optimize_array,
                                                       min_count_worker_team, max_count_worker_team,
                                                       # dummy
                                                       lambda _: Time(0)))

                c_st = None
                if index == 0:  # we are scheduling the work `start of the project`
                    c_st = assigned_parent_time  # this work should always have st = 0, so we just re-assign it

                c_st, _, exec_times = \
                    timeline.find_min_start_time_with_additional(node, worker_team, node2swork, c_st,
                                                                 assigned_parent_time, work_estimator)

                c_ft = c_st
                for node_lag, node_time in exec_times.values():
                    c_ft += node_lag + node_time

                return c_st, c_ft, worker_team

            st, ft, contractor, best_worker_team = run_contractor_search(contractors, run_with_contractor)

            inseparable_chain = node.get_inseparable_chain()
            if inseparable_chain:
                skipped_inseparable_children.update((ch.id for ch in inseparable_chain))

            # finish scheduling with time spec
            timeline.schedule(index, node, node2swork, best_worker_team, contractor,
                              st, work_spec.assigned_time, assigned_parent_time, work_estimator)

        schedule_start_time = min([swork.start_time for swork in node2swork.values() if
                                   len(swork.work_unit.worker_reqs) != 0])

        return node2swork.values(), schedule_start_time, timeline


class RandomizedTopologicalScheduler(TopologicalScheduler):
    def __init__(self, work_estimator: Optional[WorkTimeEstimator or None] = None,
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator=work_estimator)
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
