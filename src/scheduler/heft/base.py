from typing import List, Optional, Dict, Iterable

from external.estimate_time import WorkTimeEstimator
from scheduler.base import Scheduler, SchedulerType
from scheduler.heft.prioritization import prioritization
from scheduler.heft.time_computaion import calculate_working_time_cascade
from scheduler.resource.base import ResourceOptimizer
from scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from scheduler.utils.just_in_time_timeline import find_min_start_time, update_timeline, schedule, \
    create_timeline
from scheduler.utils.multi_contractor import get_best_contractor_and_worker_borders
from schemas.contractor import Contractor, get_worker_contractor_pool, WorkerContractorPool
from schemas.graph import WorkGraph, GraphNode
from schemas.schedule import Schedule
from schemas.scheduled_work import ScheduledWork
from utilities.base_opt import dichotomy_int
from utilities.validation import validate_schedule


class HEFTScheduler(Scheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, work_estimator)

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 validate: bool = False) \
            -> Schedule:
        worker_pool = get_worker_contractor_pool(contractors)

        ordered_nodes = prioritization(wg, self.work_estimator)

        schedule = Schedule.from_scheduled_works(
            self.build_scheduler(ordered_nodes, worker_pool, contractors, self.work_estimator),
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule

    def build_scheduler(self, ordered_nodes: List[GraphNode],
                        worker_pool: WorkerContractorPool, contractors: List[Contractor],
                        work_estimator: WorkTimeEstimator = None) \
            -> Iterable[ScheduledWork]:
        """
        Find optimal number of workers who ensure the nearest finish time.
        Finish time is combination of two dependencies: max finish time, max time of waiting of needed workers
        This is selected by iteration from minimum possible numbers of workers until then the finish time is decreasing
        :param contractors:
        :param work_estimator:
        :param ordered_nodes:
        :param worker_pool:
        :return:
        """
        # dict for writing parameters of completed_jobs
        node2swork: Dict[str, ScheduledWork] = {}
        # list for support the queue of workers
        timeline = create_timeline(worker_pool)
        # add to queue all available workers

        for node in reversed(ordered_nodes):  # the tasks with the highest rank will be done first
            work_unit = node.work_unit
            if node.is_inseparable_son() or node.id in node2swork:  # here
                continue

            min_count_worker_team, max_count_worker_team, contractor, workers \
                = get_best_contractor_and_worker_borders(worker_pool, contractors, work_unit.worker_reqs)

            best_worker_team = [worker.copy() for worker in workers]

            def get_finish_time(worker_team):
                return find_min_start_time(node, worker_team,
                                           timeline,
                                           node2swork) + calculate_working_time_cascade(node, worker_team,
                                                                                        work_estimator)

            self.resource_optimizer.optimize_resources(worker_pool, contractors, best_worker_team,
                                                       min_count_worker_team, max_count_worker_team, get_finish_time)

            c_ft = schedule(node, node2swork, best_worker_team, contractor, timeline, work_estimator)

            # add using resources in queue for workers
            update_timeline(c_ft, timeline, best_worker_team)

        # parallelize_local_sequence(ordered_nodes, 0, len(ordered_nodes), work_id2schedule_unit)
        # recalc_schedule(reversed(ordered_nodes), work_id2schedule_unit, worker_pool, work_estimator)

        return node2swork.values()
