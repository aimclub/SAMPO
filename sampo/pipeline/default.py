from sampo.pipeline.base import InputPipeline, SchedulePipeline
from sampo.pipeline.delegating import DelegatingScheduler
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.base import Scheduler
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.apply_queue import ApplyQueue
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.project import ScheduledProject
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.structurator import graph_restructuring
from sampo.userinput.parser.csv_parser import CSVParser
from sampo.utilities.task_name import NameMapper

import pandas as pd


class DefaultInputPipeline(InputPipeline):
    """
    Default pipeline, that help to use the framework
    """

    def __init__(self):
        self._wg: WorkGraph | pd.DataFrame | str | None = None
        self._contractors: list[Contractor] | pd.DataFrame | str | None = None
        self._work_estimator: WorkTimeEstimator = DefaultWorkEstimator()
        self._node_order: list[GraphNode] | None = None
        self._lag_optimize: LagOptimizationStrategy = LagOptimizationStrategy.NONE
        self._spec: ScheduleSpec | None = ScheduleSpec()
        self._assigned_parent_time: Time | None = Time(0)
        self._local_optimize_stack: ApplyQueue = ApplyQueue()
        self._landscape_config = LandscapeConfiguration()
        self._history: pd.DataFrame | None = None
        self._is_wg_has_full_info_about_connections: bool = False
        self._change_base_on_history: bool = False
        self._name_mapper: NameMapper | None = None

    def wg(self, wg: WorkGraph | pd.DataFrame | str,
           is_wg_has_full_info_about_connections: bool = False,
           change_base_on_history: bool = False) -> 'InputPipeline':
        """
        Mandatory argument.

        :param change_base_on_history: whether it is necessary to change project information based on connection history data
        :param is_wg_has_full_info_about_connections: does the project information contain full details of the works
        :param wg: the WorkGraph object for scheduling task
        :return: the pipeline object
        """
        self._wg = wg
        self._is_wg_has_full_info_about_connections = is_wg_has_full_info_about_connections
        self._change_base_on_history = change_base_on_history
        return self

    def contractors(self, contractors: list[Contractor] | pd.DataFrame | str) -> 'InputPipeline':
        """
        Mandatory argument.

        :param contractors: the contractors list for scheduling task
        :return: the pipeline object
        """
        if not DefaultInputPipeline._check_is_contractors_can_perform_work_graph(contractors, self._wg):
            raise NoSufficientContractorError('Contractors are not able to perform the graph of works')
        self._contractors = contractors
        return self

    def landscape(self, landscape_config: LandscapeConfiguration) -> 'InputPipeline':
        """
        Set landscape configuration

        :param landscape_config:
        :return:
        """
        self._landscape_config = landscape_config
        return self

    def name_mapper(self, name_mapper: NameMapper) -> 'InputPipeline':
        """
        Set works' name mapper
        :param name_mapper:
        :return:
        """
        self._name_mapper = name_mapper
        return self

    def history(self, history: pd.DataFrame | str) -> 'InputPipeline':
        """
        Set historical data. Mandatory method, if work graph hasn't info about links
        :param history:
        :return:
        """
        self._history = history
        return self

    def spec(self, spec: ScheduleSpec) -> 'InputPipeline':
        """
        Set specification of schedule

        :param spec:
        :return:
        """
        self._spec = spec
        return self

    def time_shift(self, time: Time) -> 'InputPipeline':
        """
        If the schedule should start at a certain time

        :param time:
        :return:
        """
        self._assigned_parent_time = time
        return self

    def lag_optimize(self, lag_optimize: LagOptimizationStrategy) -> 'InputPipeline':
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
        if isinstance(self._wg, pd.DataFrame) or isinstance(self._wg, str):
            self._wg, self._contractors = \
                CSVParser.work_graph_and_contractors(
                    works_info=CSVParser.read_graph_info(self._wg,
                                                         self._history,
                                                         self._is_wg_has_full_info_about_connections,
                                                         self._change_base_on_history),
                    contractor_info=self._contractors,
                    work_resource_estimator=self._work_estimator,
                    unique_work_names_mapper=self._name_mapper
                )
        if isinstance(scheduler, GenericScheduler):
            # if scheduler is generic, it supports injecting local optimizations
            # cache upper-layer self to another variable to get it from inner class
            s_self = self

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

        match self._lag_optimize:
            case LagOptimizationStrategy.NONE:
                wg = self._wg
                schedule, _, _, node_order = scheduler.schedule_with_cache(wg, self._contractors,
                                                                           self._landscape_config,
                                                                           self._spec,
                                                                           assigned_parent_time=self._assigned_parent_time)
                self._node_order = node_order

            case LagOptimizationStrategy.AUTO:
                # Searching the best
                wg1 = graph_restructuring(self._wg, False)
                schedule1, _, _, node_order1 = scheduler.schedule_with_cache(wg1, self._contractors,
                                                                             self._landscape_config,
                                                                             self._spec,
                                                                             assigned_parent_time=self._assigned_parent_time)
                wg2 = graph_restructuring(self._wg, True)
                schedule2, _, _, node_order2 = scheduler.schedule_with_cache(wg2, self._contractors,
                                                                             self._landscape_config,
                                                                             self._spec,
                                                                             assigned_parent_time=self._assigned_parent_time)

                if schedule1.execution_time < schedule2.execution_time:
                    self._node_order = node_order1
                    wg = wg1
                    schedule = schedule1
                else:
                    self._node_order = node_order2
                    wg = wg2
                    schedule = schedule2

            case _:
                wg = graph_restructuring(self._wg, self._lag_optimize)
                schedule, _, _, node_order = scheduler.schedule_with_cache(wg, self._contractors,
                                                                           self._landscape_config,
                                                                           self._spec,
                                                                           assigned_parent_time=self._assigned_parent_time)
                self._node_order = node_order

        return DefaultSchedulePipeline(self, wg, schedule)

    @staticmethod
    def _check_is_contractors_can_perform_work_graph(contractors: list[Contractor], wg: WorkGraph) -> bool:
        is_at_least_one_contractor_can_perform = True
        num_contractors_can_perform_node = 0

        for node in wg.nodes:
            reqs = node.work_unit.worker_reqs
            for contractor in contractors:
                offers = contractor.workers
                for req in reqs:
                    if req.min_count > offers[req.kind].count:
                        is_at_least_one_contractor_can_perform = False
                        break
                if is_at_least_one_contractor_can_perform:
                    num_contractors_can_perform_node += 1
                    break
                is_at_least_one_contractor_can_perform = True
            if num_contractors_can_perform_node == 0:
                return False
            num_contractors_can_perform_node = 0

        return True


# noinspection PyProtectedMember
class DefaultSchedulePipeline(SchedulePipeline):

    def __init__(self, s_input: DefaultInputPipeline, wg: WorkGraph, schedule: Schedule):
        self._input = s_input
        self._wg = wg
        self._worker_pool = get_worker_contractor_pool(s_input._contractors)
        self._schedule = schedule
        self._scheduled_works = {wg[swork.id]:
                                 swork for swork in schedule.to_schedule_work_dict.values()}
        self._local_optimize_stack = ApplyQueue()

    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        self._local_optimize_stack.add(optimizer.optimize,
                                       (
                                           self._input._node_order, self._input._contractors,
                                           self._input._landscape_config,
                                           self._input._spec, self._worker_pool, self._input._work_estimator,
                                           self._input._assigned_parent_time, area))
        return self

    def finish(self) -> ScheduledProject:
        processed_sworks = self._local_optimize_stack.apply(self._scheduled_works)
        schedule = Schedule.from_scheduled_works(processed_sworks.values(), self._wg)
        return ScheduledProject(self._wg, self._input._contractors, schedule)
