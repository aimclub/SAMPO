"""Base scheduling types and abstract scheduler implementation.
Базовые типы планировщиков и абстрактная реализация планировщика.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import (
    CoordinateDescentResourceOptimizer,
)
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.base_opt import dichotomy_int


class SchedulerType(Enum):
    """Enumeration of available scheduler implementations.
    Перечисление доступных реализаций планировщика.
    """

    Genetic = "genetic"
    Topological = "topological"
    HEFTAddEnd = "heft_add_end"
    HEFTAddBetween = "heft_add_between"
    LFT = "LFT"


class Scheduler(ABC):
    """Base class that implements scheduling logic.
    Базовый класс, реализующий логику планирования.
    """

    scheduler_type: SchedulerType
    resource_optimizer: ResourceOptimizer

    def __init__(
        self,
        scheduler_type: SchedulerType,
        resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(
            dichotomy_int
        ),
        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
    ):
        """Initialize the scheduler.
        Инициализирует планировщик.

        Args:
            scheduler_type: Type of scheduler implementation.
                Тип реализации планировщика.
            resource_optimizer: Optimizer for resource allocation.
                Оптимизатор распределения ресурсов.
            work_estimator: Work time estimator.
                Оценщик времени выполнения работ.
        """
        self.scheduler_type = scheduler_type
        self.resource_optimizer = resource_optimizer
        self.work_estimator = work_estimator

    def __str__(self) -> str:
        """Return scheduler type name.
        Возвращает имя типа планировщика.
        """
        return str(self.scheduler_type.name)

    def schedule(
        self,
        wg: WorkGraph,
        contractors: list[Contractor],
        spec: ScheduleSpec = ScheduleSpec(),
        validate: bool = False,
        start_time: Time = Time(0),
        timeline: Timeline | None = None,
        landscape: LandscapeConfiguration = LandscapeConfiguration(),
    ) -> list[Schedule]:
        """Run the scheduling process and return schedules.
        Запускает процесс планирования и возвращает расписания.

        Args:
            wg: Work graph to schedule.
                Граф работ для планирования.
            contractors: Available contractors.
                Доступные подрядчики.
            spec: Scheduling specification.
                Спецификация планирования.
            validate: Whether to validate the resulting schedule.
                Нужно ли проверять полученное расписание.
            start_time: Schedule start time.
                Время начала расписания.
            timeline: Optional timeline instance.
                Необязательный экземпляр временной шкалы.
            landscape: Landscape configuration.
                Конфигурация ландшафта.

        Returns:
            list[Schedule]: Generated schedules.
                Сформированные расписания.

        Raises:
            ValueError: If work graph or contractors are missing.
                ValueError: Если граф работ или подрядчики отсутствуют.
        """
        if wg is None or len(wg.nodes) == 0:
            raise ValueError("None or empty WorkGraph")
        if contractors is None or len(contractors) == 0:
            raise ValueError("None or empty contractor list")
        schedules = self.schedule_with_cache(
            wg, contractors, spec, validate, start_time, timeline, landscape
        )
        schedules = [schedule[0] for schedule in schedules]
        # print(f'Schedule exec time: {schedule.execution_time} days')
        return schedules

    @abstractmethod
    def schedule_with_cache(
        self,
        wg: WorkGraph,
        contractors: list[Contractor],
        spec: ScheduleSpec = ScheduleSpec(),
        validate: bool = False,
        assigned_parent_time: Time = Time(0),
        timeline: Timeline | None = None,
        landscape: LandscapeConfiguration = LandscapeConfiguration(),
    ) -> list[tuple[Schedule, Time, Timeline, list[GraphNode]]]:
        """Extended scheduling returning inner information.
        Расширенное планирование, возвращающее внутреннюю информацию.

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
                Schedules with metadata.
                Расписания с дополнительными данными.
        """
        ...

    @staticmethod
    def optimize_resources_using_spec(
        work_unit: WorkUnit,
        worker_team: list[Worker],
        work_spec: WorkSpec,
        optimize_lambda: Callable[[np.ndarray], None] = lambda _: None,
    ):
        """Apply a worker specification during optimization.
        Применяет спецификацию работников в процессе оптимизации.

        Args:
            work_unit: Current work unit.
                Текущая работа.
            worker_team: Worker team from the chosen contractor.
                Команда работников выбранного подрядчика.
            work_spec: Specification for the work unit.
                Спецификация для данной работы.
            optimize_lambda: Optimization callback using `optimize_array`.
                Callback оптимизации, использующий `optimize_array`.
        """
        worker_reqs = set(wr.kind for wr in work_unit.worker_reqs)
        worker_team = [worker for worker in worker_team if worker.name in worker_reqs]

        if len(work_spec.assigned_workers) == len(work_unit.worker_reqs):
            # all resources passed in spec, skipping optimize_resources step
            for worker in worker_team:
                worker.count = work_spec.assigned_workers[worker.name]
        else:
            # create optimize array to save optimizing time
            # this array contains True for positions to optimize and False otherwise
            optimize_array = None
            if work_spec.assigned_workers:
                optimize_array = []
                for worker in worker_team:
                    spec_count = work_spec.assigned_workers.get(worker.name, 0)
                    if spec_count > 0:
                        worker.count = spec_count
                        optimize_array.append(False)
                    else:
                        optimize_array.append(True)

                optimize_array = np.array(optimize_array)

            optimize_lambda(optimize_array)
