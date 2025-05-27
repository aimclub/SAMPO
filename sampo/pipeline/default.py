import pandas as pd

from sampo.generator.environment import ContractorGenerationMethod
from sampo.pipeline.base import InputPipeline, SchedulePipeline
from sampo.pipeline.delegating import DelegatingScheduler
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.pipeline.preparation import PreparationPipeline
from sampo.scheduler.base import Scheduler
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.apply_queue import ApplyQueue
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.project import ScheduledProject
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.stochastic_graph import StochasticGraph
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.structurator import graph_restructuring
from sampo.userinput.parser.csv_parser import CSVParser
from sampo.utilities.name_mapper import NameMapper, read_json


def contractors_can_perform_work_graph(contractors: list[Contractor], wg: WorkGraph) -> bool:
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


class DefaultInputPipeline(InputPipeline):
    """
    Default pipeline, that help to use the framework
    """

    def __init__(self):
        self._wg: WorkGraph | pd.DataFrame | str | None = None
        self._stochastic_wg: StochasticGraph | None = None
        self._contractors: list[Contractor] | pd.DataFrame | str | tuple[ContractorGenerationMethod, int] | None \
            = ContractorGenerationMethod.AVG, 1
        self._work_estimator: WorkTimeEstimator = DefaultWorkEstimator()
        self._node_orders: list[list[GraphNode]] | None = None
        self._lag_optimize: LagOptimizationStrategy = LagOptimizationStrategy.NONE
        self._spec: ScheduleSpec | None = ScheduleSpec()
        self._assigned_parent_time: Time | None = Time(0)
        self._local_optimize_stack: ApplyQueue = ApplyQueue()
        self._landscape_config = LandscapeConfiguration()
        self._preparation = PreparationPipeline()
        self._history: pd.DataFrame = pd.DataFrame(columns=['marker_for_glue', 'work_name', 'first_day', 'last_day',
                                                            'upper_works', 'work_name_clear_old', 'smr_name',
                                                            'work_name_clear', 'granular_smr_name'])
        self._all_connections: bool = False
        self._change_connections_info: bool = False
        self._name_mapper: NameMapper | None = None
        self.sep_wg = ';'
        self.sep_history = ';'

    def wg(self,
           wg: WorkGraph | pd.DataFrame | str,
           change_base_on_history: bool = False,
           sep: str = ';',
           all_connections: bool = False,
           change_connections_info: bool = False) -> 'InputPipeline':
        """
        Mandatory argument.

        :param change_base_on_history: whether it is necessary to change project information based on connection history data
        :param is_wg_has_full_info_about_connections: does the project information contain full details of the works
        :param wg: the WorkGraph object for scheduling task
        :param sep: separating character. It's mandatory, if you send the file path with work_info

        ATTENTION!
            If you send WorkGraph .csv or HistoryData file path, use the same separating character in
            work_info.csv as in history_data.csv and vice versa.
        """
        self._wg = wg
        self._all_connections = all_connections
        self._change_connections_info = change_connections_info
        self.sep_wg = sep
        return self

    def stochastic_wg(self, stochastic_wg: StochasticGraph):
        self._stochastic_wg = stochastic_wg
        return self

    def contractors(self, contractors: list[Contractor] | pd.DataFrame | str | tuple[ContractorGenerationMethod, int]) \
            -> 'InputPipeline':
        """
        Mandatory argument.

        :param contractors: the contractors list for scheduling task, or DataFrame with contractor info,
                            or file with contractor info, of method for contractors generation with
                            number of contractors to be generated
        :return: the pipeline object
        """
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

    def name_mapper(self, name_mapper: NameMapper | str) -> 'InputPipeline':
        """
        Set works' name mapper
        :param name_mapper:
        :return:
        """
        if isinstance(name_mapper, str):
            name_mapper = read_json(name_mapper)
        self._name_mapper = name_mapper
        return self

    def history(self, history: pd.DataFrame | str, sep: str = ';') -> 'InputPipeline':
        """
        Set historical data. Mandatory method, if work graph hasn't info about links
        :param history:
        :param sep: separating character. It's mandatory, if you send the file path with work_info

        ATTENTION!
            If you send WorkGraph .csv or HistoryData file path, use the same separating character in
            work_info.csv as in history_data.csv and vice versa.
        """
        self._history = history
        self.sep_history = sep
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

    def node_order(self, node_orders: list[list[GraphNode]]) -> 'InputPipeline':
        self._node_orders = node_orders
        return self

    def optimize_local(self, optimizer: OrderLocalOptimizer, area: range) -> 'InputPipeline':
        self._local_optimize_stack.add(optimizer.optimize, area)
        return self

    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        if isinstance(self._wg, pd.DataFrame) or isinstance(self._wg, str):
            self._wg, self._contractors = \
                CSVParser.work_graph_and_contractors(
                    works_info=CSVParser.read_graph_info(project_info=self._wg,
                                                         history_data=self._history,
                                                         sep_wg=self.sep_wg,
                                                         sep_history=self.sep_history,
                                                         name_mapper=self._name_mapper,
                                                         all_connections=self._all_connections,
                                                         change_connections_info=self._change_connections_info),
                    contractor_info=self._contractors,
                    work_resource_estimator=self._work_estimator
                )

        if not contractors_can_perform_work_graph(self._contractors, self._wg):
            raise NoSufficientContractorError('Contractors are not able to perform the graph of works')

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
                schedules = scheduler.schedule_with_cache(wg, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time)
                node_orders = [node_order for _, _, _, node_order in schedules]
                schedules = [schedule for schedule, _, _, _ in schedules]
                self._node_orders = node_orders

            case LagOptimizationStrategy.AUTO:
                # Searching the best
                wg1 = graph_restructuring(self._wg, False)
                schedules = scheduler.schedule_with_cache(wg1, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time)
                node_orders1 = [node_order for _, _, _, node_order in schedules]
                schedules1 = [schedule for schedule, _, _, _ in schedules]
                min_time1 = min([schedule.execution_time for schedule in schedules1])

                wg2 = graph_restructuring(self._wg, True)
                schedules = scheduler.schedule_with_cache(wg2, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time)
                node_orders2 = [node_order for _, _, _, node_order in schedules]
                schedules2 = [schedule for schedule, _, _, _ in schedules]
                min_time2 = min([schedule.execution_time for schedule in schedules2])

                if min_time1 < min_time2:
                    self._node_orders = node_orders1
                    wg = wg1
                    schedules = schedules1
                else:
                    self._node_orders = node_orders2
                    wg = wg2
                    schedules = schedules2

            case _:
                wg = graph_restructuring(self._wg, self._lag_optimize.value)
                schedules = scheduler.schedule_with_cache(wg, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time)
                node_orders = [node_order for _, _, _, node_order in schedules]
                schedules = [schedule for schedule, _, _, _ in schedules]
                self._node_orders = node_orders

        return DefaultSchedulePipeline(self, wg, schedules)

    def stochastic_schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        # TODO Make ability to fix all random curcumstances in stochastic wg generation,
        #  e.g. every full generation should be the same as others
        schedules = scheduler.stochastic_schedule_with_cache(self._stochastic_wg, self._contractors, self._spec,
                                                             landscape=self._landscape_config,
                                                             assigned_parent_time=self._assigned_parent_time)
        # FIXME now without multi-schedule (pareto) support
        schedule = schedules[0][0]
        stochastic_wg_realized = schedules[0][4]
        self._node_orders = [schedules[0][3]]
        # TODO WG????? wtf..
        return DefaultSchedulePipeline(self, stochastic_wg_realized, [schedule])


# noinspection PyProtectedMember
class DefaultSchedulePipeline(SchedulePipeline):

    def __init__(self, s_input: DefaultInputPipeline, wg: WorkGraph, schedules: list[Schedule]):
        self._input = s_input
        self._wg = wg
        self._worker_pool = get_worker_contractor_pool(s_input._contractors)
        self._schedules = schedules
        self._scheduled_works = [{wg[swork.id]: swork for swork in schedule.to_schedule_work_dict.values()}
                                 for schedule in schedules]
        self._local_optimize_stack = ApplyQueue()
        self._start_date = None

    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        self._local_optimize_stack.add(optimizer.optimize,
                                       self._input._contractors, self._input._landscape_config,
                                       self._input._spec, self._worker_pool, self._input._work_estimator,
                                       self._input._assigned_parent_time, area)
        return self

    def finish(self) -> list[ScheduledProject]:
        scheduled_projects = []
        for scheduled_works, node_order in zip(self._scheduled_works, self._input._node_orders):
            processed_sworks = self._local_optimize_stack.apply(scheduled_works, node_order)
            schedule = Schedule.from_scheduled_works(processed_sworks.values(), self._wg)
            scheduled_projects.append(ScheduledProject(self._input._wg, self._wg, self._input._contractors, schedule))
        return scheduled_projects

    def visualization(self, start_date: str) -> list['Visualization']:
        from sampo.utilities.visualization import Visualization
        return [Visualization.from_project(project, start_date) for project in self.finish()]
