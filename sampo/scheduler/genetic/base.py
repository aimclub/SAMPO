"""Genetic scheduling orchestrator.

Генетический оркестратор планирования.
"""

import random
from typing import Optional

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.genetic.operators import FitnessFunction, TimeFitness
from sampo.scheduler.genetic.schedule_builder import build_schedules, build_schedules_with_cache
from sampo.scheduler.genetic.converter import ScheduleGenerationScheme
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.lft.base import LFTScheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.identity import IdentityResourceOptimizer
from sampo.scheduler.resources_in_time.average_binary_search import AverageBinarySearchResourceOptimizingScheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.validation import validate_schedule


class GeneticScheduler(Scheduler):
    """Hybrid scheduler combining heuristics and genetic search.

    Гибридный планировщик, сочетающий эвристику и генетический поиск.
    """

    def __init__(self,
                 number_of_generation: Optional[int] = 50,
                 mutate_order: Optional[float | None] = None,
                 mutate_resources: Optional[float | None] = None,
                 mutate_zones: Optional[float | None] = None,
                 size_of_population: Optional[float | None] = None,
                 rand: Optional[random.Random] = None,
                 seed: Optional[float | None] = None,
                 weights: Optional[list[int] | None] = None,
                 fitness_constructor: FitnessFunction = TimeFitness(),
                 fitness_weights: tuple[int | float, ...] = (-1,),
                 scheduler_type: SchedulerType = SchedulerType.Genetic,
                 resource_optimizer: ResourceOptimizer = IdentityResourceOptimizer(),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                 optimize_resources: bool = False,
                 is_multiobjective: bool = False,
                 only_lft_initialization: bool = False):
        """Initialize genetic scheduler.

        Инициализирует генетический планировщик.

        Args:
            number_of_generation (int, optional): Number of generations.
                Количество поколений.
            mutate_order (float | None, optional): Mutation probability for order.
                Вероятность мутации порядка.
            mutate_resources (float | None, optional): Mutation probability for resources.
                Вероятность мутации ресурсов.
            mutate_zones (float | None, optional): Mutation probability for zones.
                Вероятность мутации зон.
            size_of_population (float | None, optional): Population size.
                Размер популяции.
            rand (random.Random, optional): Random generator.
                Генератор случайных чисел.
            seed (float | None, optional): Seed for random generator.
                Значение инициализации случайных чисел.
            weights (list[int] | None, optional): Weights for initial schedules.
                Веса для начальных расписаний.
            fitness_constructor (FitnessFunction): Fitness function factory.
                Фабрика функции пригодности.
            fitness_weights (tuple[int | float, ...]): Weights for fitness values.
                Веса для значений пригодности.
            scheduler_type (SchedulerType): Scheduler type.
                Тип планировщика.
            resource_optimizer (ResourceOptimizer): Optimizer for resources.
                Оптимизатор ресурсов.
            work_estimator (WorkTimeEstimator): Work time estimator.
                Оценщик времени работ.
            sgs_type (ScheduleGenerationScheme): Schedule generation scheme.
                Схема генерации расписания.
            optimize_resources (bool): Whether to optimize resources.
                Оптимизировать ли ресурсы.
            is_multiobjective (bool): Use multiobjective optimization.
                Использовать ли многокритериальную оптимизацию.
            only_lft_initialization (bool): Initialize using only LFT.
                Инициализировать только через LFT.
        """
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=resource_optimizer,
                         work_estimator=work_estimator)
        self.number_of_generation = number_of_generation
        self.mutate_order = mutate_order
        self.mutate_resources = mutate_resources
        self.mutate_zones = mutate_zones
        self.size_of_population = size_of_population
        self.rand = rand or random.Random(seed)
        self.fitness_constructor = fitness_constructor
        self.fitness_weights = fitness_weights
        self.work_estimator = work_estimator
        self.sgs_type = sgs_type

        self._optimize_resources = optimize_resources
        self._is_multiobjective = is_multiobjective
        self._weights = weights
        self._only_lft_initialization = only_lft_initialization

        self._time_border = None
        self._max_plateau_steps = None
        self._deadline = None

    def __str__(self) -> str:
        return f'GeneticScheduler[' \
               f'generations={self.number_of_generation},' \
               f'population_size={self.size_of_population},' \
               f'mutate_order={self.mutate_order},' \
               f'mutate_resources={self.mutate_resources}' \
               f']'

    def get_params(self, works_count: int) -> tuple[float, float, float, int]:
        """Get GA parameters derived from work count.

        Получает параметры ГА на основе числа работ.

        Args:
            works_count (int): Number of works in graph.
                Количество работ в графе.

        Returns:
            tuple[float, float, float, int]: Probabilities for order, resources and
                zones mutations plus population size.
                Вероятности мутаций порядка, ресурсов и зон, а также размер
                популяции.
        """
        mutate_order = self.mutate_order
        if mutate_order is None:
            mutate_order = 0.05

        mutate_resources = self.mutate_resources
        if mutate_resources is None:
            mutate_resources = 0.05

        mutate_zones = self.mutate_zones
        if mutate_zones is None:
            mutate_zones = 0.05

        size_of_population = self.size_of_population
        if size_of_population is None:
            if works_count < 300:
                size_of_population = 50
            elif 1500 > works_count >= 300:
                size_of_population = 100
            else:
                size_of_population = works_count // 25
        return mutate_order, mutate_resources, mutate_zones, size_of_population

    def set_time_border(self, time_border: int) -> None:
        """Define execution time border for evaluation.

        Устанавливает ограничение времени выполнения для оценки.

        Args:
            time_border (int): Maximum execution time.
                Максимальное время выполнения.
        """
        self._time_border = time_border

    def set_max_plateau_steps(self, max_plateau_steps: int) -> None:
        """Define plateau limit for GA evolution.

        Задает предел плато для эволюции ГА.

        Args:
            max_plateau_steps (int): Steps without improvement.
                Шаги без улучшений.
        """
        self._max_plateau_steps = max_plateau_steps

    def set_deadline(self, deadline: Time) -> None:
        """Set the project deadline.

        Устанавливает дедлайн проекта.

        Args:
            deadline (Time): Finish time limit.
                Ограничение по времени завершения.
        """
        self._deadline = deadline

    def set_weights(self, weights: list[int]) -> None:
        """Assign weights for initial schedules.

        Назначает веса для начальных расписаний.

        Args:
            weights (list[int]): Importance values.
                Значения важности.
        """
        self._weights = weights

    def set_optimize_resources(self, optimize_resources: bool) -> None:
        """Toggle resource optimization.

        Переключает оптимизацию ресурсов.

        Args:
            optimize_resources (bool): Enable optimization.
                Включить оптимизацию.
        """
        self._optimize_resources = optimize_resources

    def set_is_multiobjective(self, is_multiobjective: bool) -> None:
        """Toggle multiobjective mode.

        Переключает режим многокритериальности.

        Args:
            is_multiobjective (bool): Enable multiobjective search.
                Включить многокритериальный поиск.
        """
        self._is_multiobjective = is_multiobjective

    def set_only_lft_initialization(self, only_lft_initialization: bool) -> None:
        """Use only LFT initialization.

        Использовать только инициализацию LFT.

        Args:
            only_lft_initialization (bool): Flag for LFT-only start.
                Флаг запуска только через LFT.
        """
        self._only_lft_initialization = only_lft_initialization

    @staticmethod
    def generate_first_population(wg: WorkGraph,
                                  contractors: list[Contractor],
                                  landscape: LandscapeConfiguration = LandscapeConfiguration(),
                                  spec: ScheduleSpec = ScheduleSpec(),
                                  work_estimator: WorkTimeEstimator | None = None,
                                  deadline: Time | None = None,
                                  weights=None):
        """Create initial population using heuristic schedulers.

        Создает начальную популяцию с помощью эвристических планировщиков.

        Args:
            wg (WorkGraph): Graph of works.
                Граф работ.
            contractors (list[Contractor]): Available contractors.
                Доступные подрядчики.
            landscape (LandscapeConfiguration): Landscape configuration.
                Конфигурация ландшафта.
            spec (ScheduleSpec): Scheduling specification.
                Спецификация расписания.
            work_estimator (WorkTimeEstimator | None): Time estimator.
                Оценщик времени.
            deadline (Time | None): Optional project deadline.
                Необязательный дедлайн проекта.
            weights (list[int] | None): Importance weights.
                Веса важности.

        Returns:
            dict[str, tuple[Schedule | None, list[GraphNode] | None, ScheduleSpec | None, int]]:
                Mapping of heuristic name to initial schedule data.
                Сопоставление имени эвристики с данными начального расписания.
        """

        if weights is None:
            weights = [2, 2, 2, 1, 1, 1, 1]

        schedule, _, _, node_order = LFTScheduler(work_estimator=work_estimator).schedule_with_cache(wg, contractors,
                                                                                                     spec,
                                                                                                     landscape=landscape)[0]
        init_lft_schedule = (schedule, node_order, spec)

        def init_k_schedule(scheduler_class, k) -> tuple[Schedule | None, list[GraphNode] | None, ScheduleSpec | None]:
            try:
                schedule, _, _, node_order = (scheduler_class(work_estimator=work_estimator,
                                                              resource_optimizer=AverageReqResourceOptimizer(k))
                                              .schedule_with_cache(wg, contractors, spec, landscape=landscape))[0]
                return schedule, node_order, spec
            except NoSufficientContractorError:
                return None, None, None

        if deadline is None:
            def init_schedule(scheduler_class) -> tuple[Schedule | None, list[GraphNode] | None, ScheduleSpec | None]:
                try:
                    schedule, _, _, node_order = (scheduler_class(work_estimator=work_estimator)
                                                  .schedule_with_cache(wg, contractors, spec, landscape=landscape))[0]
                    return schedule, node_order, spec
                except NoSufficientContractorError:
                    return None, None, None

        else:
            def init_schedule(scheduler_class) -> tuple[Schedule | None, list[GraphNode] | None, ScheduleSpec | None]:
                try:
                    (schedule, _, _, node_order), modified_spec = AverageBinarySearchResourceOptimizingScheduler(
                        scheduler_class(work_estimator=work_estimator)
                    ).schedule_with_cache(wg, contractors, deadline, spec, landscape=landscape)
                    return schedule, node_order, modified_spec
                except NoSufficientContractorError:
                    return None, None, None

        return {
            "lft": (*init_lft_schedule, weights[0]),
            "heft_end": (*init_schedule(HEFTScheduler), weights[1]),
            "heft_between": (*init_schedule(HEFTBetweenScheduler), weights[2]),
            "12.5%": (*init_k_schedule(HEFTScheduler, 8), weights[3]),
            "25%": (*init_k_schedule(HEFTScheduler, 4), weights[4]),
            "75%": (*init_k_schedule(HEFTScheduler, 4 / 3), weights[5]),
            "87.5%": (*init_k_schedule(HEFTScheduler, 8 / 7), weights[6])
        }

    def upgrade_pop(self,
                    wg: WorkGraph,
                    contractors: list[Contractor],
                    pop: list[ChromosomeType],
                    spec: ScheduleSpec = ScheduleSpec(),
                    assigned_parent_time: Time = Time(0),
                    timeline: Timeline | None = None,
                    landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
        """Evolve an existing population.

        Развивает существующую популяцию.

        Args:
            wg (WorkGraph): Work graph.
                Граф работ.
            contractors (list[Contractor]): Contractors list.
                Список подрядчиков.
            pop (list[ChromosomeType]): Current population.
                Текущая популяция.
            spec (ScheduleSpec): Scheduling specification.
                Спецификация расписания.
            assigned_parent_time (Time): Start time of parent schedule.
                Время начала родительского расписания.
            timeline (Timeline | None): Existing timeline.
                Существующая временная шкала.
            landscape (LandscapeConfiguration): Landscape configuration.
                Конфигурация ландшафта.

        Returns:
            list[ChromosomeType]: Updated population.
                Обновленная популяция.
        """
        mutate_order, mutate_resources, mutate_zones, size_of_population = self.get_params(wg.vertex_count)
        deadline = None if self._optimize_resources else self._deadline

        _, new_pop = build_schedules_with_cache(wg,
                                                contractors,
                                                size_of_population,
                                                self.number_of_generation,
                                                mutate_order,
                                                mutate_resources,
                                                mutate_zones,
                                                {},
                                                self.rand,
                                                spec,
                                                self._weights,
                                                pop,
                                                landscape,
                                                self.fitness_constructor,
                                                self.fitness_weights,
                                                self.work_estimator,
                                                self.sgs_type,
                                                assigned_parent_time,
                                                timeline,
                                                self._time_border,
                                                self._max_plateau_steps,
                                                self._optimize_resources,
                                                deadline,
                                                self._only_lft_initialization,
                                                self._is_multiobjective)
        return new_pop

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None,
                            landscape: LandscapeConfiguration = LandscapeConfiguration()) \
            -> list[tuple[Schedule, Time, Timeline, list[GraphNode]]]:
        """Build schedules and return cached state.

        Строит расписания и возвращает их кэшированное состояние.

        Args:
            wg (WorkGraph): Work graph.
                Граф работ.
            contractors (list[Contractor]): Contractors list.
                Список подрядчиков.
            spec (ScheduleSpec): Scheduling specification.
                Спецификация расписания.
            validate (bool): Whether to validate schedules.
                Нужно ли валидировать расписания.
            assigned_parent_time (Time): Start time of parent schedule.
                Время начала родительского расписания.
            timeline (Timeline | None): Existing timeline.
                Существующая временная шкала.
            landscape (LandscapeConfiguration): Landscape configuration.
                Конфигурация ландшафта.

        Returns:
            list[tuple[Schedule, Time, Timeline, list[GraphNode]]]: Generated schedules
                with metadata.
                Сгенерированные расписания с метаданными.
        """
        init_schedules = GeneticScheduler.generate_first_population(wg, contractors, landscape, spec,
                                                                    self.work_estimator, self._deadline, self._weights)

        mutate_order, mutate_resources, mutate_zones, size_of_population = self.get_params(wg.vertex_count)
        deadline = None if self._optimize_resources else self._deadline

        schedules = build_schedules(wg,
                                    contractors,
                                    size_of_population,
                                    self.number_of_generation,
                                    mutate_order,
                                    mutate_resources,
                                    mutate_zones,
                                    init_schedules,
                                    self.rand,
                                    spec,
                                    self._weights,
                                    None,
                                    landscape,
                                    self.fitness_constructor,
                                    self.fitness_weights,
                                    self.work_estimator,
                                    self.sgs_type,
                                    assigned_parent_time,
                                    timeline,
                                    self._time_border,
                                    self._max_plateau_steps,
                                    self._optimize_resources,
                                    deadline,
                                    self._only_lft_initialization,
                                    self._is_multiobjective)
        schedules = [
            (Schedule.from_scheduled_works(scheduled_works.values(), wg), schedule_start_time, timeline, order_nodes)
            for scheduled_works, schedule_start_time, timeline, order_nodes in schedules]

        if validate:
            for schedule, *_ in schedules:
                validate_schedule(schedule, wg, contractors)

        return schedules
