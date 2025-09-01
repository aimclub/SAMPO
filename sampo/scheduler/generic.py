"""Generic scheduler implementation with pluggable strategies.
Реализация универсального планировщика с подключаемыми стратегиями.
"""

from typing import Callable, Iterable, Type

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils import (
    WorkerContractorPool,
    get_head_nodes_with_connections_mappings,
    get_worker_contractor_pool,
)
from sampo.scheduler.utils.multi_contractor import (
    get_worker_borders,
    run_contractor_search,
)
from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import DefaultWorkEstimator, WorkTimeEstimator
from sampo.utilities.validation import validate_schedule


# TODO Кажется, это не работает - лаги не учитываются
def get_finish_time_default(
    node,
    worker_team,
    node2swork,
    spec,
    assigned_parent_time,
    timeline,
    work_estimator,
) -> Time:
    """Estimate finish time using default method.
    Оценивает время завершения стандартным методом.

    Args:
        node: Current graph node.
            Текущая вершина графа.
        worker_team: Worker team assigned to the node.
            Команда работников, назначенная на вершину.
        node2swork: Mapping of nodes to scheduled works.
            Отображение вершин в запланированные работы.
        spec: Work specification.
            Спецификация работы.
        assigned_parent_time: Parent start time.
            Время начала родителя.
        timeline: Timeline instance.
            Экземпляр временной шкалы.
        work_estimator: Work time estimator.
            Оценщик времени выполнения работ.

    Returns:
        Time: Estimated finish time.
            Оценка времени завершения.
    """
    return timeline.find_min_start_time(
        node,
        worker_team,
        node2swork,
        spec,
        assigned_parent_time,
        work_estimator,
    ) + calculate_working_time_cascade(
        node,
        worker_team,
        work_estimator,
    )  # TODO Кажется, это не работает - лаги не учитываются


PRIORITIZATION_F = Callable[
    [list[GraphNode], dict[str, set[str]], dict[str, set[str]], WorkTimeEstimator],
    list[GraphNode],
]
RESOURCE_OPTIMIZE_F = Callable[
    [
        GraphNode,
        list[Contractor],
        WorkSpec,
        WorkerContractorPool,
        dict[GraphNode, ScheduledWork],
        Time,
        Timeline,
        WorkTimeEstimator,
    ],
    tuple[Time, Time, Contractor, list[Worker]],
]


class GenericScheduler(Scheduler):
    """Universal scheduler with customizable strategies.
    Универсальный планировщик с настраиваемыми стратегиями.
    """

    def __init__(
        self,
        scheduler_type: SchedulerType,
        resource_optimizer: ResourceOptimizer,
        timeline_type: Type,
        prioritization_f: PRIORITIZATION_F,
        optimize_resources_f: RESOURCE_OPTIMIZE_F,
        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
    ):
        """Create a generic scheduler instance.
        Создает экземпляр универсального планировщика.

        Args:
            scheduler_type: Type of scheduler implementation.
                Тип реализации планировщика.
            resource_optimizer: Resource optimization strategy.
                Стратегия оптимизации ресурсов.
            timeline_type: Timeline class to use.
                Используемый класс временной шкалы.
            prioritization_f: Node prioritization function.
                Функция приоритизации узлов.
            optimize_resources_f: Resource optimization function.
                Функция оптимизации ресурсов.
            work_estimator: Work time estimator.
                Оценщик времени выполнения работ.
        """
        super().__init__(scheduler_type, resource_optimizer, work_estimator)
        self._timeline_type = timeline_type
        self.prioritization = prioritization_f
        self.optimize_resources = optimize_resources_f

    def get_default_res_opt_function(
        self, get_finish_time=get_finish_time_default
    ) -> Callable[
        [
            GraphNode,
            list[Contractor],
            WorkSpec,
            WorkerContractorPool,
            dict[GraphNode, ScheduledWork],
            Time,
            Timeline,
            WorkTimeEstimator,
        ],
        tuple[Time, Time, Contractor, list[Worker]],
    ]:
        """Return default resource optimization function.
        Возвращает функцию оптимизации ресурсов по умолчанию.

        Args:
            get_finish_time: Function estimating finish time.
                Функция, оценивающая время завершения.

        Returns:
            Callable: Resource optimization function for scheduler construction.
                Callable: Функция оптимизации ресурсов для создания планировщика.
        """

        def optimize_resources_def(
            node: GraphNode,
            contractors: list[Contractor],
            spec: WorkSpec,
            worker_pool: WorkerContractorPool,
            node2swork: dict[GraphNode, ScheduledWork],
            assigned_parent_time: Time,
            timeline: Timeline,
            work_estimator: WorkTimeEstimator,
        ):
            def ft_getter(worker_team) -> Time:
                return get_finish_time(
                    node,
                    worker_team,
                    node2swork,
                    spec,
                    assigned_parent_time,
                    timeline,
                    work_estimator,
                )

            def run_with_contractor(
                contractor: Contractor,
            ) -> tuple[Time, Time, list[Worker]]:
                min_count_worker_team, max_count_worker_team, workers = (
                    get_worker_borders(
                        worker_pool, contractor, node.work_unit.worker_reqs
                    )
                )

                if len(workers) != len(node.work_unit.worker_reqs):
                    return assigned_parent_time, Time.inf(), []

                workers = [worker.copy() for worker in workers]

                # apply worker team spec
                self.optimize_resources_using_spec(
                    node.work_unit,
                    workers,
                    spec,
                    lambda optimize_array: self.resource_optimizer.optimize_resources(
                        worker_pool,
                        workers,
                        optimize_array,
                        min_count_worker_team,
                        max_count_worker_team,
                        ft_getter,
                    ),
                )

                c_st, c_ft, _ = timeline.find_min_start_time_with_additional(
                    node,
                    workers,
                    node2swork,
                    spec,
                    None,
                    assigned_parent_time,
                    work_estimator,
                )
                return c_st, c_ft, workers

            return run_contractor_search(contractors, run_with_contractor)

        return optimize_resources_def

    def schedule_with_cache(
        self,
        wg: WorkGraph,
        contractors: list[Contractor],
        spec: ScheduleSpec = ScheduleSpec(),
        validate: bool = False,
        assigned_parent_time: Time = Time(0),
        timeline: Timeline | None = None,
        landscape: LandscapeConfiguration() = LandscapeConfiguration(),
    ) -> list[tuple[Schedule, Time, Timeline, list[GraphNode]]]:
        """Schedule graph and return additional info.
        Планирует граф и возвращает дополнительную информацию.

        Args:
            wg: Work graph to schedule.
                Граф работ для планирования.
            contractors: Available contractors.
                Доступные подрядчики.
            spec: Scheduling specification.
                Спецификация планирования.
            validate: Whether to validate the resulting schedule.
                Нужно ли проверять полученное расписание.
            assigned_parent_time: Start time of the parent schedule.
                Время начала родительского расписания.
            timeline: Optional timeline instance.
                Необязательный экземпляр временной шкалы.
            landscape: Landscape configuration.
                Конфигурация ландшафта.

        Returns:
            list[tuple[Schedule, Time, Timeline, list[GraphNode]]]:
                Schedules with execution context.
                Расписания с контекстом выполнения.
        """
        # get head nodes with mappings
        head_nodes, node_id2parent_ids, node_id2child_ids = (
            get_head_nodes_with_connections_mappings(wg)
        )

        ordered_nodes = self.prioritization(
            head_nodes, node_id2parent_ids, node_id2child_ids, self.work_estimator
        )

        schedule, schedule_start_time, timeline = self.build_scheduler(
            ordered_nodes,
            contractors,
            landscape,
            spec,
            self.work_estimator,
            assigned_parent_time,
            timeline,
        )
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg,
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return [(schedule, schedule_start_time, timeline, ordered_nodes)]

    def build_scheduler(
        self,
        ordered_nodes: list[GraphNode],
        contractors: list[Contractor],
        landscape: LandscapeConfiguration = LandscapeConfiguration(),
        spec: ScheduleSpec = ScheduleSpec(),
        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
        assigned_parent_time: Time = Time(0),
        timeline: Timeline | None = None,
    ) -> tuple[Iterable[ScheduledWork], Time, Timeline]:
        """Construct schedule from ordered nodes.
        Формирует расписание из упорядоченных узлов.

        Args:
            ordered_nodes: Nodes ordered for scheduling.
                Узлы, упорядоченные для планирования.
            contractors: Available contractors.
                Доступные подрядчики.
            landscape: Landscape configuration.
                Конфигурация ландшафта.
            spec: Specification for scheduling.
                Спецификация для планирования.
            work_estimator: Work time estimator.
                Оценщик времени выполнения работ.
            assigned_parent_time: Start time of the whole schedule.
                Время начала всего расписания.
            timeline: Optional timeline instance.
                Необязательный экземпляр временной шкалы.

        Returns:
            tuple[Iterable[ScheduledWork], Time, Timeline]:
                Scheduled works, start time, and timeline.
                Запланированные работы, время начала и временная шкала.
        """
        worker_pool = get_worker_contractor_pool(contractors)
        # dict for writing parameters of completed_jobs
        node2swork: dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(worker_pool, landscape)

        for index, node in enumerate(ordered_nodes):
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)

            start_time, finish_time, contractor, best_worker_team = (
                self.optimize_resources(
                    node,
                    contractors,
                    work_spec,
                    worker_pool,
                    node2swork,
                    assigned_parent_time,
                    timeline,
                    work_estimator,
                )
            )

            # we are scheduling the work `start of the project`
            if index == 0:
                # this work should always have start_time = 0, so we just re-assign it
                start_time = assigned_parent_time
                finish_time += start_time

            if (
                index == len(ordered_nodes) - 1
            ):  # we are scheduling the work `end of the project`
                finish_time, finalizing_zones = timeline.zone_timeline.finish_statuses()
                start_time = max(start_time, finish_time)

            # apply work to scheduling
            timeline.schedule(
                node,
                node2swork,
                best_worker_team,
                contractor,
                work_spec,
                start_time,
                work_spec.assigned_time,
                assigned_parent_time,
                work_estimator,
            )

            if (
                index == len(ordered_nodes) - 1
            ):  # we are scheduling the work `end of the project`
                node2swork[node].zones_pre = finalizing_zones

        return node2swork.values(), assigned_parent_time, timeline
