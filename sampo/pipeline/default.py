from sampo.pipeline.base import InputPipeline, SchedulePipeline
from sampo.pipeline.delegating import DelegatingScheduler
from sampo.scheduler.base import Scheduler
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.apply_queue import ApplyQueue
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.structurator import graph_restructuring


class DefaultInputPipeline(InputPipeline):

    def __init__(self):
        self._wg = None
        self._contractors = None
        self._work_estimator = None
        self._node_order = None
        self._lag_optimize = None
        self._spec = ScheduleSpec()
        self._assigned_parent_time = Time(0)
        self._local_optimize_stack = ApplyQueue()

    def wg(self, wg: WorkGraph) -> 'InputPipeline':
        """
        Mandatory argument.

        :param wg: the WorkGraph object for scheduling task
        :return: the pipeline object
        """
        self._wg = wg
        return self

    def contractors(self, contractors: list[Contractor]) -> 'InputPipeline':
        """
        Mandatory argument.
        :param contractors: the contractors list for scheduling task
        :return: the pipeline object
        """
        self._contractors = contractors
        return self

    def spec(self, spec: ScheduleSpec) -> 'InputPipeline':
        self._spec = spec
        return self

    def time_shift(self, time: Time) -> 'InputPipeline':
        self._assigned_parent_time = time
        return self

    def lag_optimize(self, lag_optimize: bool) -> 'InputPipeline':
        """
        Mandatory argument. Shows should graph be lag-optimized or not.
        If not defined, pipeline should search the best variant of this argument in result.

        :param lag_optimize:
        :return: the pipeline object
        """
        self._lag_optimize = lag_optimize
        return self

    def work_estimator(self, work_estimator: WorkTimeEstimator) -> 'InputPipeline':
        self._work_estimator = work_estimator
        return self

    def node_order(self, node_order: list[GraphNode]) -> 'InputPipeline':
        self._node_order = node_order
        return self

    def optimize_local(self, optimizer: OrderLocalOptimizer, area: range) -> 'InputPipeline':
        self._local_optimize_stack.add(optimizer.optimize, (area,))
        return self

    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        if isinstance(scheduler, GenericScheduler):
            # if scheduler is generic, it supports injecting local optimizations
            s_self = self  # cache upper-layer self to another variable to get it from inner class

            class LocalOptimizedScheduler(DelegatingScheduler):

                def __init__(self, delegate: GenericScheduler):
                    super().__init__(delegate)

                def delegate_prioritization(self, orig_prioritization):
                    def prioritization(wg: WorkGraph, work_estimator: WorkTimeEstimator):
                        # call delegate's prioritization and apply local optimizations
                        return s_self._local_optimize_stack.apply(orig_prioritization(wg, work_estimator))

                    return prioritization

            scheduler = LocalOptimizedScheduler(scheduler)
        elif not self._local_optimize_stack.empty():
            print('Trying to apply local optimizations to non-generic scheduler, ignoring it')

        if self._lag_optimize is None:
            # Searching the best
            wg = graph_restructuring(self._wg, False)
            schedule1, _, _, node_order1 = scheduler.schedule_with_cache(wg, self._contractors, self._spec,
                                                                         assigned_parent_time=self._assigned_parent_time)
            wg = graph_restructuring(self._wg, True)
            schedule2, _, _, node_order2 = scheduler.schedule_with_cache(wg, self._contractors, self._spec,
                                                                         assigned_parent_time=self._assigned_parent_time)

            if schedule1.execution_time < schedule2.execution_time:
                self._node_order = node_order1
                schedule = schedule1
            else:
                self._node_order = node_order2
                schedule = schedule2

        else:
            wg = graph_restructuring(self._wg, self._lag_optimize)
            schedule, _, _, node_order = scheduler.schedule_with_cache(wg, self._contractors, self._spec,
                                                                       assigned_parent_time=self._assigned_parent_time)
            self._node_order = node_order

        return DefaultSchedulePipeline(self, wg, schedule)


# noinspection PyProtectedMember
class DefaultSchedulePipeline(SchedulePipeline):

    def __init__(self, s_input: DefaultInputPipeline, wg: WorkGraph, schedule: Schedule):
        self._input = s_input
        self._wg = wg
        self._worker_pool = get_worker_contractor_pool(s_input._contractors)
        self._schedule = schedule
        self._scheduled_works = {wg[swork.work_unit.id]:
                                 swork for swork in schedule.to_schedule_work_dict.values()}
        self._local_optimize_stack = ApplyQueue()

    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        self._local_optimize_stack.add(optimizer.optimize,
                                       (self._input._node_order, self._input._contractors, self._input._spec,
                                        self._worker_pool, self._input._work_estimator,
                                        self._input._assigned_parent_time, area))
        return self

    def finish(self) -> Schedule:
        processed_sworks = self._local_optimize_stack.apply(self._scheduled_works)
        return Schedule.from_scheduled_works(processed_sworks.values(), self._wg)
