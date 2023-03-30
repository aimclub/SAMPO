from typing import List, Optional, Type

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.heft.time_computaion import calculate_working_time_cascade
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.scheduler.utils.multi_contractor import get_worker_borders, run_contractor_search
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.base_opt import dichotomy_int


class HEFTScheduler(GenericScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 timeline_type: Type = JustInTimeTimeline,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, timeline_type, prioritization,
                         self.optimize_resources_default, work_estimator)
        self._timeline_type = timeline_type

    def optimize_resources_default(self, node: GraphNode, contractors: list[Contractor], work_spec: WorkSpec,
                                   worker_pool: WorkerContractorPool, node2swork: dict[GraphNode, ScheduledWork],
                                   assigned_parent_time: Time, timeline: Timeline, work_estimator: WorkTimeEstimator):
        def run_with_contractor(contractor: Contractor) -> tuple[Time, Time, List[Worker]]:
            min_count_worker_team, max_count_worker_team, workers \
                = get_worker_borders(worker_pool, contractor, node.work_unit.worker_reqs)

            if len(workers) != len(node.work_unit.worker_reqs):
                return Time(0), Time.inf(), []

            workers = [worker.copy() for worker in workers]

            def get_finish_time(worker_team):
                return timeline.find_min_start_time(node, worker_team, node2swork,
                                                    assigned_parent_time, work_estimator) \
                    + calculate_working_time_cascade(node, worker_team, work_estimator)

            # apply worker team spec
            self.optimize_resources_using_spec(node.work_unit, workers, work_spec,
                                               lambda optimize_array: self.resource_optimizer.optimize_resources(
                                                   worker_pool, workers,
                                                   optimize_array,
                                                   min_count_worker_team, max_count_worker_team,
                                                   get_finish_time))

            # c_st = timeline.find_min_start_time(node, workers, node2swork, assigned_parent_time, work_estimator)
            # c_ft = c_st + calculate_working_time_cascade(node, workers, work_estimator)
            c_st, c_ft, _ = timeline.find_min_start_time_with_additional(node, workers, node2swork, None,
                                                                         assigned_parent_time, work_estimator)
            return c_st, c_ft, workers

        return run_contractor_search(contractors, run_with_contractor)


class HEFTBetweenScheduler(HEFTScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddBetween,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, MomentumTimeline, work_estimator)
