from typing import Type, List, Callable, Iterable

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.validation import validate_schedule


class GenericScheduler(Scheduler):

    def __init__(self,
                 scheduler_type: SchedulerType,
                 resource_optimizer: ResourceOptimizer,
                 timeline_type: Type,
                 prioritization_f: Callable[[WorkGraph, WorkTimeEstimator], list[GraphNode]],
                 optimize_resources_f: Callable[[GraphNode, list[Contractor], WorkSpec, WorkerContractorPool,
                                                 dict[GraphNode, ScheduledWork], Time, Timeline, WorkTimeEstimator],
                                                tuple[Time, Time, Contractor, list[Worker]]],
                 work_estimator: WorkTimeEstimator | None = None):
        super().__init__(scheduler_type, resource_optimizer, work_estimator)
        self._timeline_type = timeline_type
        self.prioritization = prioritization_f
        self.optimize_resources = optimize_resources_f

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
        ordered_nodes = self.prioritization(wg, self.work_estimator)

        schedule, schedule_start_time, timeline = \
            self.build_scheduler(ordered_nodes, contractors, spec, self.work_estimator, assigned_parent_time, timeline)
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline, ordered_nodes

    def build_scheduler(self,
                        ordered_nodes: List[GraphNode],
                        contractors: List[Contractor],
                        spec: ScheduleSpec,
                        work_estimator: WorkTimeEstimator = None,
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, Timeline]:
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
        node2swork: dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(ordered_nodes, contractors, worker_pool)

        for index, node in enumerate(reversed(ordered_nodes)):  # the tasks with the highest rank will be done first
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)

            st, ft, contractor, best_worker_team = self.optimize_resources(node, contractors, work_spec, worker_pool,
                                                                           node2swork, assigned_parent_time,
                                                                           timeline, work_estimator)

            if index == 0:  # we are scheduling the work `start of the project`
                st = assigned_parent_time  # this work should always have st = 0, so we just re-assign it
                ft += st

            # apply work to scheduling
            timeline.schedule(index, node, node2swork, best_worker_team, contractor,
                              st, work_spec.assigned_time, assigned_parent_time, work_estimator)

        schedule_start_time = min([swork.start_time for swork in node2swork.values() if
                                   len(swork.work_unit.worker_reqs) != 0])

        return node2swork.values(), schedule_start_time, timeline
