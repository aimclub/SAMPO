"""Default pipelines for scheduling operations.

Стандартные конвейеры для планирования операций.
"""

from __future__ import annotations

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
from sampo.utilities.priority import check_and_correct_priorities
from sampo.schemas.apply_queue import ApplyQueue
from sampo.schemas.contractor import Contractor
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
from sampo.utilities.name_mapper import NameMapper, read_json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from sampo.utilities.visualization import Visualization


def contractors_can_perform_work_graph(
    contractors: list[Contractor],
    wg: WorkGraph,
) -> bool:
    """Check if each work node has an eligible contractor.

    Проверяет, имеет ли каждый узел работ подходящего подрядчика.

    Args:
        contractors (list[Contractor]): Available contractors.
            Доступные подрядчики.
        wg (WorkGraph): Work graph to evaluate.
            Граф работ для оценки.

    Returns:
        bool: True if each node can be performed by at least one contractor.
        bool: True, если каждый узел может быть выполнен хотя бы одним
            подрядчиком.
    """
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
    """Default pipeline simplifying framework usage.

    Стандартный конвейер, упрощающий использование фреймворка.
    """

    def __init__(self) -> None:
        """Initialize pipeline with default components.

        Инициализирует конвейер с компонентами по умолчанию.
        """
        self._wg: WorkGraph | pd.DataFrame | str | None = None
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
        self.sep_wg = ','
        self.sep_history = ','

    def wg(self,
           wg: WorkGraph | pd.DataFrame | str,
           change_base_on_history: bool = False,
           sep: str = ',',
           all_connections: bool = False,
           change_connections_info: bool = False) -> 'InputPipeline':
        """Set work graph for scheduling.

        Устанавливает граф работ для планирования.

        Args:
            wg (WorkGraph | pd.DataFrame | str): Work graph or path/dataframe.
                Граф работ либо путь/таблица.
            change_base_on_history (bool): Modify info using history data.
                Изменять информацию проекта на основе истории.
            sep (str): Separator for CSV files.
                Разделитель для CSV-файлов.
            all_connections (bool): Whether graph contains all connections.
                Содержит ли граф всю информацию о связях.
            change_connections_info (bool): Update connections from history.
                Обновлять связи из истории.

        Warning:
            If file paths are provided, use the same separator in
            ``work_info.csv`` and ``history_data.csv``.
            При передаче путей используйте одинаковый разделитель в
            ``work_info.csv`` и ``history_data.csv``.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._wg = wg
        self._all_connections = all_connections
        self._change_connections_info = change_connections_info
        self.sep_wg = sep
        return self

    def contractors(self, contractors: list[Contractor] | pd.DataFrame | str | tuple[ContractorGenerationMethod, int]) \
            -> 'InputPipeline':
        """Set contractors for scheduling.

        Устанавливает подрядчиков для планирования.

        Args:
            contractors (list[Contractor] | pd.DataFrame | str | tuple[ContractorGenerationMethod, int]):
                Contractors list, dataframe, file path, or generation method and count.
                Список подрядчиков, DataFrame, путь к файлу или метод генерации и количество.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._contractors = contractors
        return self

    def landscape(self, landscape_config: LandscapeConfiguration) -> 'InputPipeline':
        """Set landscape configuration.

        Устанавливает конфигурацию ландшафта.

        Args:
            landscape_config (LandscapeConfiguration): Landscape settings.
                Настройки ландшафта.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._landscape_config = landscape_config
        return self

    def name_mapper(self, name_mapper: NameMapper | str) -> 'InputPipeline':
        """Set works' name mapper.

        Устанавливает преобразователь имён работ.

        Args:
            name_mapper (NameMapper | str): Mapper object or path to JSON.
                Объект преобразователя или путь к JSON.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        if isinstance(name_mapper, str):
            name_mapper = read_json(name_mapper)
        self._name_mapper = name_mapper
        return self

    def history(self, history: pd.DataFrame | str, sep: str = ';') -> 'InputPipeline':
        """Set historical data for link recovery.

        Устанавливает исторические данные для восстановления связей.

        Args:
            history (pd.DataFrame | str): History dataframe or path.
                DataFrame с историей или путь к файлу.
            sep (str): Separator used in CSV files.
                Разделитель, применяемый в CSV-файлах.

        Warning:
            Use the same separator in ``work_info.csv`` and ``history_data.csv``.
            Используйте одинаковый разделитель в ``work_info.csv`` и ``history_data.csv``.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._history = history
        self.sep_history = sep
        return self

    def spec(self, spec: ScheduleSpec) -> 'InputPipeline':
        """Set schedule specification.

        Устанавливает спецификацию расписания.

        Args:
            spec (ScheduleSpec): Scheduling constraints.
                Ограничения планирования.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._spec = spec
        return self

    def time_shift(self, time: Time) -> 'InputPipeline':
        """Assign start time for the schedule.

        Назначает время начала расписания.

        Args:
            time (Time): Desired start time.
                Желаемое время начала.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._assigned_parent_time = time
        return self

    def lag_optimize(self, lag_optimize: LagOptimizationStrategy) -> 'InputPipeline':
        """Specify lag optimization strategy.

        Указывает стратегию оптимизации временных разрывов.

        Args:
            lag_optimize (LagOptimizationStrategy): Selected optimization strategy.
                Выбранная стратегия оптимизации.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._lag_optimize = lag_optimize
        return self

    def work_estimator(self, work_estimator: WorkTimeEstimator) -> 'InputPipeline':
        """Set work time estimator.

        Устанавливает оценщик времени работ.

        Args:
            work_estimator (WorkTimeEstimator): Estimator instance.
                Экземпляр оценщика.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._work_estimator = work_estimator
        return self

    def node_order(self, node_orders: list[list[GraphNode]]) -> 'InputPipeline':
        """Define custom order for groups of nodes.

        Определяет пользовательский порядок групп узлов.

        Args:
            node_orders (list[list[GraphNode]]): Ordered node groups.
                Упорядоченные группы узлов.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._node_orders = node_orders
        return self

    def optimize_local(self, optimizer: OrderLocalOptimizer, area: range) -> 'InputPipeline':
        """Add local optimizer to the pipeline.

        Добавляет локальный оптимизатор в конвейер.

        Args:
            optimizer (OrderLocalOptimizer): Optimizer to apply.
                Применяемый оптимизатор.
            area (range): Range of nodes affected.
                Диапазон затрагиваемых узлов.

        Returns:
            InputPipeline: Pipeline object for chaining.
            InputPipeline: Объект конвейера для цепочек.
        """
        self._local_optimize_stack.add(optimizer.optimize, area)
        return self

    def schedule(self, scheduler: Scheduler, validate: bool = False) -> 'SchedulePipeline':
        """Build schedule using provided scheduler.

        Строит расписание с использованием указанного планировщика.

        Args:
            scheduler (Scheduler): Scheduling algorithm.
                Алгоритм планирования.
            validate (bool): Whether to validate resulting schedule.
                Проверять ли итоговое расписание.

        Returns:
            SchedulePipeline: Pipeline producing the schedule.
            SchedulePipeline: Конвейер, формирующий расписание.
        """
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

        check_and_correct_priorities(self._wg)

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
                    def prioritization(head_nodes: list[GraphNode],
                                       node_id2parent_ids: dict[str, set[str]],
                                       node_id2child_ids: dict[str, set[str]],
                                       work_estimator: WorkTimeEstimator):
                        # call delegate's prioritization and apply local optimizations
                        return s_self._local_optimize_stack.apply(
                            orig_prioritization(head_nodes, node_id2parent_ids, node_id2child_ids, work_estimator)
                        )

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
                                                          assigned_parent_time=self._assigned_parent_time,
                                                          validate=validate)
                node_orders = [node_order for _, _, _, node_order in schedules]
                schedules = [schedule for schedule, _, _, _ in schedules]
                self._node_orders = node_orders

            case LagOptimizationStrategy.AUTO:
                # Searching the best
                wg1 = graph_restructuring(self._wg, False)
                schedules = scheduler.schedule_with_cache(wg1, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time,
                                                          validate=validate)
                node_orders1 = [node_order for _, _, _, node_order in schedules]
                schedules1 = [schedule for schedule, _, _, _ in schedules]
                min_time1 = min([schedule.execution_time for schedule in schedules1])

                wg2 = graph_restructuring(self._wg, True)
                schedules = scheduler.schedule_with_cache(wg2, self._contractors,
                                                          self._spec,
                                                          landscape=self._landscape_config,
                                                          assigned_parent_time=self._assigned_parent_time,
                                                          validate=validate)
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
                                                          assigned_parent_time=self._assigned_parent_time,
                                                          validate=validate)
                node_orders = [node_order for _, _, _, node_order in schedules]
                schedules = [schedule for schedule, _, _, _ in schedules]
                self._node_orders = node_orders

        return DefaultSchedulePipeline(self, wg, schedules)


# noinspection PyProtectedMember
class DefaultSchedulePipeline(SchedulePipeline):
    """Pipeline for processing generated schedules.

    Конвейер для обработки полученных расписаний.
    """

    def __init__(self, s_input: DefaultInputPipeline, wg: WorkGraph, schedules: list[Schedule]) -> None:
        """Initialize schedule pipeline.

        Инициализирует конвейер расписаний.

        Args:
            s_input (DefaultInputPipeline): Source input pipeline.
                Исходный конвейер ввода.
            wg (WorkGraph): Work graph used for scheduling.
                Используемый граф работ.
            schedules (list[Schedule]): Generated schedules.
                Список сгенерированных расписаний.
        """
        self._input = s_input
        self._wg = wg
        self._worker_pool = get_worker_contractor_pool(s_input._contractors)
        self._schedules = schedules
        self._scheduled_works = [
            {wg[swork.id]: swork for swork in schedule.to_schedule_work_dict.values()}
            for schedule in schedules
        ]
        self._local_optimize_stack = ApplyQueue()
        self._start_date = None

    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        """Add local optimizer for schedule stage.

        Добавляет локальный оптимизатор для стадии расписания.

        Args:
            optimizer (ScheduleLocalOptimizer): Optimizer to apply.
                Применяемый оптимизатор.
            area (range): Range of works affected.
                Диапазон затрагиваемых работ.

        Returns:
            SchedulePipeline: Pipeline object for chaining.
            SchedulePipeline: Объект конвейера для цепочек.
        """
        self._local_optimize_stack.add(
            optimizer.optimize,
            self._input._contractors,
            self._input._landscape_config,
            self._input._spec,
            self._worker_pool,
            self._input._work_estimator,
            self._input._assigned_parent_time,
            area,
        )
        return self

    def finish(self) -> list[ScheduledProject]:
        """Finalize scheduling and return projects.

        Завершает планирование и возвращает проекты.

        Returns:
            list[ScheduledProject]: Final scheduled projects.
            list[ScheduledProject]: Итоговые запланированные проекты.
        """
        scheduled_projects = []
        for scheduled_works, node_order in zip(self._scheduled_works, self._input._node_orders):
            processed_sworks = self._local_optimize_stack.apply(scheduled_works, node_order)
            schedule = Schedule.from_scheduled_works(processed_sworks.values(), self._wg)
            scheduled_projects.append(
                ScheduledProject(self._input._wg, self._wg, self._input._contractors, schedule)
            )
        return scheduled_projects

    def visualization(self, start_date: str) -> list[Visualization]:
        """Create visualizations for scheduled projects.

        Создаёт визуализации запланированных проектов.

        Args:
            start_date (str): Start date for visualization.
                Дата начала для визуализации.

        Returns:
            list[Visualization]: Generated visualizations.
            list[Visualization]: Созданные визуализации.
        """
        from sampo.utilities.visualization import Visualization

        return [Visualization.from_project(project, start_date) for project in self.finish()]
