from typing import List, Optional, Dict, Iterable

from sampo.metrics.resources_in_time.base import ResourceOptimizer
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.heft.time_computaion import calculate_working_time_cascade
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.scheduler.utils.momentum_timeline import create_timeline, find_min_start_time, schedule_with_time_spec
from sampo.scheduler.utils.multi_contractor import get_best_contractor_and_worker_borders
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.base_opt import dichotomy_int
from sampo.utilities.validation import validate_schedule


class HEFTBetweenScheduler(HEFTScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddBetween,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, work_estimator)

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 validate: bool = False) \
            -> Schedule:
        ordered_nodes = prioritization(wg, self.work_estimator)

        schedule = Schedule.from_scheduled_works(
            self.build_scheduler(ordered_nodes, contractors, spec, self.work_estimator),
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule

    def build_scheduler(self, ordered_nodes: List[GraphNode],
                        contractors: List[Contractor],
                        spec: ScheduleSpec,
                        work_estimator: WorkTimeEstimator = None) \
            -> Iterable[ScheduledWork]:
        """
        Find optimal number of workers who ensure the nearest finish time.
        Finish time is combination of two dependencies: max finish time, max time of waiting of needed workers
        This is selected by iteration from minimum possible numbers of workers until then the finish time is decreasing
        :param ordered_nodes:
        :param contractors:
        :param spec: spec for current scheduling
        :param work_estimator:
        :return:
        """
        worker_pool = get_worker_contractor_pool(contractors)
        # dict for writing parameters of completed_jobs
        node2swork: Dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        timeline = create_timeline(ordered_nodes, contractors)
        # add to queue all available workers

        for index, node in enumerate(reversed(ordered_nodes)):  # the tasks with the highest rank will be done first
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)
            if node in node2swork:  # here
                continue

            inseparable_chain = node.get_inseparable_chain() if node.get_inseparable_chain() is not None else [node]

            min_count_worker_team, max_count_worker_team, contractor, workers \
                = get_best_contractor_and_worker_borders(worker_pool, contractors, work_unit.worker_reqs)

            best_worker_team = [worker.copy() for worker in workers]

            def get_finish_time(worker_team):
                return find_min_start_time(timeline[contractor.id], node, node2swork, inseparable_chain,
                                           best_worker_team, work_estimator)[0] \
                       + calculate_working_time_cascade(node, worker_team, work_estimator)

            # apply worker team spec
            self.optimize_resources_using_spec(work_unit, best_worker_team, work_spec,
                                               lambda optimize_array: self.resource_optimizer.optimize_resources(
                                                   worker_pool, best_worker_team,
                                                   optimize_array,
                                                   min_count_worker_team, max_count_worker_team,
                                                   get_finish_time))

            # finish scheduling with time spec
            schedule_with_time_spec(index, node, node2swork, inseparable_chain, timeline, best_worker_team, contractor,
                                    work_spec.assigned_time, work_estimator)

        # parallelize_local_sequence(ordered_nodes, 0, len(ordered_nodes), node2swork)
        # recalc_schedule(reversed(ordered_nodes), node2swork, agents, work_estimator)

        return node2swork.values()
