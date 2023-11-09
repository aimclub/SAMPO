from typing import Type, Callable, Iterable

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.heft.time_computaion import calculate_working_time_cascade
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils.multi_contractor import run_contractor_search, get_worker_borders
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.validation import validate_schedule


# TODO Кажется, это не работает - лаги не учитываются
def get_finish_time_default(node, worker_team, node2swork, spec, assigned_parent_time, timeline, work_estimator):
    return timeline.find_min_start_time(node, worker_team, node2swork, spec,
                                        assigned_parent_time, work_estimator) \
        + calculate_working_time_cascade(node, worker_team,
                                         work_estimator)  # TODO Кажется, это не работает - лаги не учитываются


class GenericScheduler(Scheduler):
    """
    Implementation of a universal scheme of scheduler.
    It's parametrized by prioritization function and optimization resource function.
    It constructs the end schedulers.
    """

    def __init__(self,
                 scheduler_type: SchedulerType,
                 resource_optimizer: ResourceOptimizer,
                 timeline_type: Type,
                 prioritization_f: Callable[[WorkGraph, WorkTimeEstimator], list[GraphNode]],
                 optimize_resources_f: Callable[[GraphNode, list[Contractor], WorkSpec, WorkerContractorPool,
                                                 dict[GraphNode, ScheduledWork], Time, Timeline, WorkTimeEstimator],
                                                 tuple[Time, Time, Contractor, list[Worker]]],
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, resource_optimizer, work_estimator)
        self._timeline_type = timeline_type
        self.prioritization = prioritization_f
        self.optimize_resources = optimize_resources_f

    def get_default_res_opt_function(self, get_finish_time=get_finish_time_default) \
            -> Callable[[GraphNode, list[Contractor], WorkSpec, WorkerContractorPool,
                         dict[GraphNode, ScheduledWork], Time, Timeline, WorkTimeEstimator],
                         tuple[Time, Time, Contractor, list[Worker]]]:

        def optimize_resources_def(node: GraphNode, contractors: list[Contractor], spec: WorkSpec,
                                   worker_pool: WorkerContractorPool, node2swork: dict[GraphNode, ScheduledWork],
                                   assigned_parent_time: Time, timeline: Timeline, work_estimator: WorkTimeEstimator):
            def run_with_contractor(contractor: Contractor) -> tuple[Time, Time, list[Worker]]:
                min_count_worker_team, max_count_worker_team, workers \
                    = get_worker_borders(worker_pool, contractor, node.work_unit.worker_reqs)

                if len(workers) != len(node.work_unit.worker_reqs):
                    return Time(0), Time.inf(), []

                workers = [worker.copy() for worker in workers]

                def ft_getter(worker_team):
                    return get_finish_time(node, worker_team, node2swork, spec,
                                           assigned_parent_time, timeline, work_estimator)

                # apply worker team spec
                self.optimize_resources_using_spec(node.work_unit, workers, spec,
                                                   lambda optimize_array: self.resource_optimizer.optimize_resources(
                                                       worker_pool, workers,
                                                       optimize_array,
                                                       min_count_worker_team, max_count_worker_team,
                                                       ft_getter))

                c_st, c_ft, _ = timeline.find_min_start_time_with_additional(node, workers, node2swork, spec, None,
                                                                             assigned_parent_time, work_estimator)
                return c_st, c_ft, workers

            return run_contractor_search(contractors, run_with_contractor)

        return optimize_resources_def

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            landscape: LandscapeConfiguration() = LandscapeConfiguration(),
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
        ordered_nodes = self.prioritization(wg, self.work_estimator)

        schedule, schedule_start_time, timeline = \
            self.build_scheduler(wg, ordered_nodes, contractors, landscape, spec, self.work_estimator,
                                 assigned_parent_time, timeline)
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline, ordered_nodes

    def build_scheduler(self,
                        wg: WorkGraph,
                        ordered_nodes: list[GraphNode],
                        contractors: list[Contractor],
                        landscape: LandscapeConfiguration = LandscapeConfiguration(),
                        spec: ScheduleSpec = ScheduleSpec(),
                        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, Timeline]:
        """
        Find optimal number of workers who ensure the nearest finish time.
        Finish time is combination of two dependencies: max finish time, max time of waiting of needed workers
        This is selected by iteration from minimum possible numbers of workers until then the finish time is decreasing

        :param landscape:
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
            timeline = self._timeline_type(worker_pool, landscape)

        for index, node in enumerate(reversed(ordered_nodes)):  # the tasks with the highest rank will be done first
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)

            start_time, finish_time, contractor, best_worker_team = self.optimize_resources(node, contractors,
                                                                                            work_spec, worker_pool,
                                                                                            node2swork,
                                                                                            assigned_parent_time,
                                                                                            timeline, work_estimator)

            # we are scheduling the work `start of the project`
            if index == 0:
                # this work should always have start_time = 0, so we just re-assign it
                start_time = assigned_parent_time
                finish_time += start_time

            # apply work to scheduling
            timeline.schedule(node, node2swork, best_worker_team, contractor, work_spec,
                              start_time, work_spec.assigned_time, assigned_parent_time, work_estimator)

        return node2swork.values(), assigned_parent_time, timeline
