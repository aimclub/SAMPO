from sampo.pipeline.base import InputPipeline, SchedulePipeline
from sampo.scheduler.base import Scheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule


class DefaultInputPipeline(InputPipeline):

    def __init__(self):
        self._wg = None
        self._contractors = None

    def wg(self, wg: WorkGraph) -> 'InputPipeline':
        self._wg = wg
        return self

    def contractors(self, contractors: list[Contractor]) -> 'InputPipeline':
        self._contractors = contractors
        return self

    def optimize_local(self, optimizer: OrderLocalOptimizer, area: slice) -> 'InputPipeline':
        pass

    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        schedule, _, _, node_order = scheduler.schedule_with_cache(self._wg, self._contractors)
        return DefaultSchedulePipeline(self._wg, self._contractors, node_order, schedule)


class DefaultSchedulePipeline(SchedulePipeline):

    def __init__(self, wg: WorkGraph, contractors: list[Contractor], node_order: list[GraphNode], schedule: Schedule):
        self._wg = wg
        self._contractors = contractors
        self._node_order = node_order
        self._schedule = schedule
        self._scheduled_works = schedule.to_schedule_work_dict

    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: slice) -> 'SchedulePipeline':
        self._schedule = optimizer.optimize(self._node_order, self._scheduled_works, area)
        return self

    def finish(self) -> Schedule:
        return Schedule.from_scheduled_works(self._scheduled_works.values(), self._wg)
