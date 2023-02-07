from typing import List, Optional, Dict, Iterable, Type

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.heft.time_computaion import calculate_working_time_cascade
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.scheduler.utils.multi_contractor import get_worker_borders, run_contractor_search
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.base_opt import dichotomy_int
from sampo.utilities.validation import validate_schedule


class HEFTScheduler(Scheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 timeline_type: Type = JustInTimeTimeline,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, work_estimator)
        self._timeline_type = timeline_type

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline]:
        ordered_nodes = prioritization(wg, self.work_estimator)

        schedule, schedule_start_time, timeline = \
            self.build_scheduler(ordered_nodes, contractors, spec, self.work_estimator, assigned_parent_time, timeline)
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline

    def build_scheduler(self,
                        ordered_nodes: List[GraphNode],
                        contractors: List[Contractor],
                        spec: ScheduleSpec,
                        work_estimator: WorkTimeEstimator = None,
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, JustInTimeTimeline]:
        """
        Find optimal number of workers who ensure the nearest finish time.
        Finish time is combination of two dependencies: max finish time, max time of waiting of needed workers
        This is selected by iteration from minimum possible numbers of workers until then the finish time is decreasing

        :param contractors:
        :param spec: spec for current scheduling
        :param ordered_nodes:
        :param timeline: the previous used timeline can be specified to handle previously scheduled works
        :param assigned_parent_time: start time of the whole schedule(time shift)
        :param work_estimator:
        :return:
        """
        worker_pool = get_worker_contractor_pool(contractors)
        # dict for writing parameters of completed_jobs
        node2swork: Dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(ordered_nodes, contractors, worker_pool)

        for index, node in enumerate(reversed(ordered_nodes)):  # the tasks with the highest rank will be done first
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)
            # if node in node2swork:  # here
            #     continue

            def run_with_contractor(contractor: Contractor) -> tuple[Time, Time, List[Worker]]:
                min_count_worker_team, max_count_worker_team, workers \
                    = get_worker_borders(worker_pool, contractor, work_unit.worker_reqs)

                if len(workers) != len(work_unit.worker_reqs):
                    return Time(0), Time.inf(), []

                workers = [worker.copy() for worker in workers]

                def get_finish_time(worker_team):
                    return timeline.find_min_start_time(node, worker_team, node2swork,
                                                        assigned_parent_time, work_estimator) \
                           + calculate_working_time_cascade(node, worker_team, work_estimator)

                # apply worker team spec
                self.optimize_resources_using_spec(work_unit, workers, work_spec,
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

            st, ft, contractor, best_worker_team = run_contractor_search(contractors, run_with_contractor)

            if index == 0:  # we are scheduling the work `start of the project`
                st = assigned_parent_time  # this work should always have st = 0, so we just re-assign it
                ft += st

            # apply work to scheduling
            timeline.schedule(index, node, node2swork, best_worker_team, contractor,
                              st, work_spec.assigned_time, assigned_parent_time, work_estimator)

        # parallelize_local_sequence(ordered_nodes, 0, len(ordered_nodes), work_id2schedule_unit)
        # recalc_schedule(reversed(ordered_nodes), work_id2schedule_unit, worker_pool, work_estimator)

        schedule_start_time = min([swork.start_time for swork in node2swork.values() if
                                   len(swork.work_unit.worker_reqs) != 0])

        return node2swork.values(), schedule_start_time, timeline


class HEFTBetweenScheduler(HEFTScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddBetween,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, MomentumTimeline, work_estimator)
