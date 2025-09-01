"""Genetic algorithm operators and fitness functions.

Операторы и функции пригодности для генетического алгоритма.
"""

import math
import random
from copy import deepcopy
from operator import attrgetter
from typing import Callable, Iterable

import numpy as np
from deap import base, tools
from deap.base import Toolbox

from sampo.api.genetic_api import ChromosomeType, FitnessFunction, Individual
from sampo.base import SAMPO
from sampo.scheduler.genetic.converter import (convert_schedule_to_chromosome, convert_chromosome_to_schedule,
                                               ScheduleGenerationScheme)
from sampo.scheduler.lft.base import RandomizedLFTScheduler
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler
from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.resource_usage import resources_peaks_sum, resources_costs_sum, resources_sum


class TimeFitness(FitnessFunction):
    """Fitness based on finish time.

    Фитнес, основанный на времени завершения.
    """

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int | float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (schedule.execution_time.value,)


class SumOfResourcesPeaksFitness(FitnessFunction):
    """Fitness from sum of resource peaks.

    Фитнес по сумме пиков использования ресурсов.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (resources_peaks_sum(schedule, self._resources_names),)


class SumOfResourcesFitness(FitnessFunction):
    """Fitness from total resource usage.

    Фитнес по суммарному использованию ресурсов.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (resources_sum(schedule, self._resources_names),)


class TimeWithResourcesFitness(FitnessFunction):
    """Fitness considering time and resource set.

    Фитнес с учетом времени и набора ресурсов.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (schedule.execution_time.value + resources_peaks_sum(schedule, self._resources_names),)


class DeadlineResourcesFitness(FitnessFunction):
    """Resource fitness with deadline constraint.

    Фитнес по ресурсам с ограничением по дедлайну.
    """

    def __init__(self,
                 deadline: Time,
                 resources_names: Iterable[str] | None = None):
        self._deadline = deadline
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int | float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (resources_peaks_sum(schedule, self._resources_names) \
                * max(1.0, schedule.execution_time.value / self._deadline.value),)


class DeadlineCostFitness(FitnessFunction):
    """Cost fitness with deadline constraint.

    Фитнес по стоимости с ограничением по дедлайну.
    """

    def __init__(self,
                 deadline: Time,
                 resources_names: Iterable[str] | None = None):
        self._deadline = deadline
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int | float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (resources_costs_sum(schedule, self._resources_names) \
                * max(1.0, schedule.execution_time.value / self._deadline.value),)


class TimeAndResourcesFitness(FitnessFunction):
    """Bi-objective fitness of time and resource peaks.

    Двухцелевой фитнес по времени и пикам ресурсов.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int, int]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return Time.inf().value, Time.inf().value
        return schedule.execution_time.value, resources_peaks_sum(schedule, self._resources_names)


def init_toolbox(wg: WorkGraph,
                 contractors: list[Contractor],
                 worker_pool: WorkerContractorPool,
                 landscape: LandscapeConfiguration,
                 index2node: dict[int, GraphNode],
                 work_id2index: dict[str, int],
                 worker_name2index: dict[str, int],
                 index2contractor_obj: dict[int, Contractor],
                 index2zone: dict[int, str],
                 init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                 mut_order_pb: float,
                 mut_res_pb: float,
                 mut_zone_pb: float,
                 statuses_available: int,
                 selection_size: int,
                 rand: random.Random,
                 spec: ScheduleSpec,
                 worker_pool_indices: dict[int, dict[int, Worker]],
                 contractor2index: dict[str, int],
                 contractor_borders: np.ndarray,
                 node_indices: list[int],
                 priorities: list[int],
                 parents: dict[int, set[int]],
                 children: dict[int, set[int]],
                 resources_border: np.ndarray,
                 assigned_parent_time: Time = Time(0),
                 fitness_weights: tuple[int | float, ...] = (-1,),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                 only_lft_initialization: bool = False,
                 is_multiobjective: bool = False) -> base.Toolbox:
    """Create toolbox with genetic operators.

    Создает набор инструментов с генетическими операторами.

    Returns:
        base.Toolbox: Configured toolbox for GA.
            Настроенный набор инструментов для ГА.
    """
    toolbox = base.Toolbox()
    toolbox.register('register_individual_constructor', register_individual_constructor, toolbox=toolbox)
    toolbox.register_individual_constructor(fitness_weights)
    # generate chromosome
    toolbox.register('generate_chromosome', generate_chromosome, wg=wg, contractors=contractors,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape)

    # create population
    toolbox.register('population', generate_chromosomes, wg=wg, contractors=contractors,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape,
                     only_lft_initialization=only_lft_initialization, toolbox=toolbox)
    # selection
    selection = tools.selNSGA2 if is_multiobjective else select_new_population
    toolbox.register('select', selection, k=selection_size)
    # combined crossover
    toolbox.register('mate', mate, rand=rand, toolbox=toolbox, priorities=priorities)
    # combined mutation
    toolbox.register('mutate', mutate, order_mutpb=mut_order_pb, res_mutpb=mut_res_pb, zone_mutpb=mut_zone_pb,
                     rand=rand, parents=parents, children=children, resources_border=resources_border,
                     statuses_available=statuses_available, priorities=priorities)
    # crossover for order
    toolbox.register('mate_order', mate_scheduling_order, rand=rand, toolbox=toolbox, priorities=priorities)
    # mutation for order
    toolbox.register('mutate_order', mutate_scheduling_order, mutpb=mut_order_pb, rand=rand, parents=parents,
                     children=children, priorities=priorities)
    # crossover for resources
    toolbox.register('mate_resources', mate_resources, rand=rand, toolbox=toolbox)
    # mutation for resources
    toolbox.register('mutate_resources', mutate_resources, resources_border=resources_border,
                     mutpb=mut_res_pb, rand=rand)
    # mutation for resource borders
    toolbox.register('mutate_resource_borders', mutate_resource_borders, contractor_borders=contractor_borders,
                     mutpb=mut_res_pb, rand=rand)
    toolbox.register('mate_post_zones', mate_for_zones, rand=rand, toolbox=toolbox)
    toolbox.register('mutate_post_zones', mutate_for_zones, rand=rand, mutpb=mut_zone_pb,
                     statuses_available=landscape.zone_config.statuses.statuses_available())

    toolbox.register('validate', is_chromosome_correct, node_indices=node_indices, parents=parents,
                     contractor_borders=contractor_borders, index2node=index2node)
    toolbox.register('schedule_to_chromosome', convert_schedule_to_chromosome,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     landscape=landscape)
    toolbox.register('evaluate_chromosome', evaluate, wg=wg, toolbox=toolbox)
    toolbox.register('chromosome_to_schedule', convert_chromosome_to_schedule, worker_pool=worker_pool,
                     index2node=index2node, index2contractor=index2contractor_obj,
                     worker_pool_indices=worker_pool_indices, assigned_parent_time=assigned_parent_time,
                     work_estimator=work_estimator, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, index2zone=index2zone,
                     landscape=landscape, sgs_type=sgs_type)
    toolbox.register('copy_individual', copy_individual, toolbox=toolbox)

    return toolbox


def evaluate(chromosome: ChromosomeType, wg: WorkGraph, toolbox: Toolbox) -> Schedule | None:
    """Evaluate chromosome to schedule if valid.

    Оценивает хромосому в расписание, если она корректна.

    Args:
        chromosome (ChromosomeType): Chromosome to evaluate.
            Хромосома для оценки.
        wg (WorkGraph): Work graph.
            Граф работ.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.

    Returns:
        Schedule | None: Built schedule or ``None`` when invalid.
            Построенное расписание или ``None`` при некорректности.
    """
    if toolbox.validate(chromosome):
        sworks = toolbox.chromosome_to_schedule(chromosome)[0]
        return Schedule.from_scheduled_works(sworks.values(), wg)
    else:
        return None


def register_individual_constructor(fitness_weights: tuple[int | float, ...], toolbox: base.Toolbox) -> None:
    """Register individual type with custom fitness.

    Регистрирует тип индивида с пользовательским фитнесом.

    Args:
        fitness_weights (tuple[int | float, ...]): Fitness weights.
            Веса функции пригодности.
        toolbox (base.Toolbox): Target toolbox.
            Целевой набор инструментов.
    """

    class IndividualFitness(base.Fitness):
        weights = fitness_weights

    toolbox.register('Individual', Individual.prepare(IndividualFitness))


def copy_individual(ind: Individual, toolbox: Toolbox) -> Individual:
    """Deep copy individual.

    Глубоко копирует индивида.

    Args:
        ind (Individual): Individual to copy.
            Индивид для копирования.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.

    Returns:
        Individual: Copied individual.
            Скопированный индивид.
    """
    return toolbox.Individual(
        (ind[0].copy(), ind[1].copy(), ind[2].copy(), deepcopy(ind[3]), ind[4].copy())
    )


def generate_chromosomes(n: int,
                         wg: WorkGraph,
                         contractors: list[Contractor],
                         spec: ScheduleSpec,
                         work_id2index: dict[str, int],
                         worker_name2index: dict[str, int],
                         contractor2index: dict[str, int],
                         contractor_borders: np.ndarray,
                         init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                         rand: random.Random,
                         toolbox: Toolbox,
                         work_estimator: WorkTimeEstimator = None,
                         landscape: LandscapeConfiguration = LandscapeConfiguration(),
                         only_lft_initialization: bool = False) -> list[ChromosomeType]:
    """Generate a list of chromosomes.

    Генерирует список хромосом.

    Args:
        n (int): Number of chromosomes.
            Количество хромосом.
        wg (WorkGraph): Work graph.
            Граф работ.
        contractors (list[Contractor]): Contractors list.
            Список подрядчиков.
        spec (ScheduleSpec): Scheduling specification.
            Спецификация расписания.
        work_id2index (dict[str, int]): Work ID to index map.
            Сопоставление идентификаторов работ индексам.
        worker_name2index (dict[str, int]): Worker name to index map.
            Сопоставление имен рабочих индексам.
        contractor2index (dict[str, int]): Contractor ID to index map.
            Сопоставление идентификаторов подрядчиков индексам.
        contractor_borders (np.ndarray): Contractor capacities.
            Вместимости подрядчиков.
        init_chromosomes (dict[str, tuple[ChromosomeType, float, ScheduleSpec]]):
            Predefined chromosomes with weights.
            Предопределенные хромосомы с весами.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        work_estimator (WorkTimeEstimator | None): Time estimator.
            Оценщик времени.
        landscape (LandscapeConfiguration): Landscape configuration.
            Конфигурация ландшафта.
        only_lft_initialization (bool): Use only LFT initialization.
            Использовать только LFT-инициализацию.

    Returns:
        list[ChromosomeType]: Generated chromosomes.
            Сгенерированные хромосомы.
    """

    def randomized_init(is_topological: bool = False) -> ChromosomeType:
        if is_topological:
            seed = int(rand.random() * 1000000)
            schedule, _, _, node_order = RandomizedTopologicalScheduler(work_estimator,
                                                                        seed).schedule_with_cache(wg, contractors, spec,
                                                                                                  landscape=landscape)[0]
        else:
            schedule, _, _, node_order = RandomizedLFTScheduler(work_estimator=work_estimator,
                                                                rand=rand).schedule_with_cache(wg, contractors, spec,
                                                                                               landscape=landscape)[0]
        return convert_schedule_to_chromosome(work_id2index, worker_name2index, contractor2index, contractor_borders,
                                              schedule, spec, landscape, node_order)

    if only_lft_initialization:
        chromosomes = [toolbox.Individual(randomized_init(is_topological=False)) for _ in range(n - 1)]
        chromosomes.append(toolbox.Individual(init_chromosomes['lft'][0]))
        return chromosomes

    count_for_specified_types = (n // 3) // len(init_chromosomes)
    count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
    weights = [importance for _, importance, _ in init_chromosomes.values()]
    sum_of_weights = sum(weights)
    weights = [weight / sum_of_weights for weight in weights]

    counts = [math.ceil(count_for_specified_types * weight) for weight in weights]
    sum_counts_for_specified_types = sum(counts)

    count_for_topological = n // 2 - sum_counts_for_specified_types
    count_for_topological = count_for_topological if count_for_topological > 0 else 1
    counts.append(count_for_topological)

    count_for_rand_lft = n - count_for_topological - sum_counts_for_specified_types
    count_for_rand_lft = count_for_rand_lft if count_for_rand_lft > 0 else 1
    counts.append(count_for_rand_lft)

    chromosome_types = rand.sample(list(init_chromosomes.keys()) + ['topological', 'rand_lft'], k=n, counts=counts)

    chromosomes = []

    for generated_type in chromosome_types:
        match generated_type:
            case 'topological':
                ind = randomized_init(is_topological=True)
            case 'rand_lft':
                ind = randomized_init(is_topological=False)
            case _:
                ind = init_chromosomes[generated_type][0]

        if not toolbox.validate(ind):
            SAMPO.logger.warn('HELP')

        ind = toolbox.Individual(ind)
        chromosomes.append(ind)

    return chromosomes[:n]


def generate_chromosome(wg: WorkGraph,
                        contractors: list[Contractor],
                        work_id2index: dict[str, int],
                        worker_name2index: dict[str, int],
                        contractor2index: dict[str, int],
                        contractor_borders: np.ndarray,
                        init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                        spec: ScheduleSpec,
                        rand: random.Random,
                        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                        landscape: LandscapeConfiguration = LandscapeConfiguration()) -> ChromosomeType:
    """Generate a single valid chromosome.

    Генерирует одну валидную хромосому.

    Uses HEFT and randomized topological orders to respect dependencies.

    Использует порядок HEFT и случайные топологические сортировки для
    соблюдения зависимостей.
    """

    def randomized_init() -> ChromosomeType:
        schedule, _, _, node_order = RandomizedTopologicalScheduler(work_estimator,
                                                  int(rand.random() * 1000000)) \
            .schedule_with_cache(wg, contractors, spec, landscape=landscape)[0]
        return convert_schedule_to_chromosome(work_id2index, worker_name2index,
                                              contractor2index, contractor_borders,
                                              schedule, spec, landscape, node_order)

    chance = rand.random()
    if chance < 0.2:
        chromosome = init_chromosomes['heft_end'][0]
    elif chance < 0.4:
        chromosome = init_chromosomes['heft_between'][0]
    elif chance < 0.5:
        chromosome = init_chromosomes['12.5%'][0]
    elif chance < 0.6:
        chromosome = init_chromosomes['25%'][0]
    elif chance < 0.7:
        chromosome = init_chromosomes['75%'][0]
    elif chance < 0.8:
        chromosome = init_chromosomes['87.5%'][0]
    else:
        chromosome = randomized_init()

    return chromosome


def select_new_population(population: list[Individual], k: int) -> list[Individual]:
    """Select top individuals by fitness.

    Выбирает лучших индивидов по пригодности.

    Args:
        population (list[Individual]): Population to select from.
            Популяция для отбора.
        k (int): Number of individuals to select.
            Количество выбираемых индивидов.

    Returns:
        list[Individual]: Selected individuals.
            Отобранные индивиды.
    """
    population = sorted(population, key=attrgetter('fitness'), reverse=True)
    return population[:k]


def is_chromosome_correct(ind: Individual, node_indices: list[int], parents: dict[int, set[int]],
                          contractor_borders: np.ndarray, index2node: dict[int, GraphNode]) -> bool:
    """Check order and contractor borders for correctness.

    Проверяет корректность порядка работ и границ подрядчиков.
    """
    return is_chromosome_order_correct(ind, parents, index2node) and \
        is_chromosome_contractors_correct(ind, node_indices, contractor_borders)


def is_chromosome_order_correct(ind: Individual, parents: dict[int, set[int]], index2node: dict[int, GraphNode]) -> bool:
    """Verify that work order is topologically valid.

    Проверяет, что порядок работ топологически верен.
    """
    work_order = ind[0]
    used = set()
    for work_index in work_order:
        used.add(work_index)
        if not parents[work_index].issubset(used):
            # logger.error(f'Order validation failed: {work_order}')
            return False

        # validate priorities
        work_node = index2node[work_index]
        if any(index2node[parent].work_unit.priority > work_node.work_unit.priority for parent in parents[work_index]):
            # SAMPO.logger.error(f'Order validation failed')
            return False
    return True


def is_chromosome_contractors_correct(ind: Individual, work_indices: Iterable[int],
                                      contractor_borders: np.ndarray) -> bool:
    """Ensure contractors can supply assigned workers.

    Убеждается, что подрядчики обеспечивают назначенных рабочих.
    """
    if not work_indices:
        return True
    resources = ind[1][work_indices]
    # sort resource part of chromosome by contractor ids
    resources = resources[resources[:, -1].argsort()]
    # get unique contractors and indexes where they start
    contractors, indexes = np.unique(resources[:, -1], return_index=True)
    # get borders of received contractors from chromosome
    chromosome_borders = ind[2][contractors]
    # split resources to get parts grouped by contractor parts
    res_grouped_by_contractor = np.split(resources[:, :-1], indexes[1:])
    # for each grouped parts take maximum for each resource
    max_of_res_by_contractor = np.array([r.max(axis=0) for r in res_grouped_by_contractor])
    return (max_of_res_by_contractor <= chromosome_borders).all() and \
        (chromosome_borders <= contractor_borders[contractors]).all()


def get_order_part(order: np.ndarray, other_order: np.ndarray) -> np.ndarray:
    """Extract new order fragment from second parent.

    Извлекает новый фрагмент порядка из второго родителя.
    """
    order = set(order)
    return np.array([node for node in other_order if node not in order])


def mate_scheduling_order(ind1: Individual, ind2: Individual, rand: random.Random,
                          toolbox: Toolbox, priorities: np.ndarray, copy: bool = True) -> tuple[Individual, Individual]:
    """Two-point crossover for work order.

    Двухточечный кроссовер порядка работ.

    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.
        copy (bool): Copy individuals before mating.
            Копировать индивидов перед скрещиванием.

    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    """
    child1, child2 = (toolbox.copy_individual(ind1), toolbox.copy_individual(ind2)) if copy else (ind1, ind2)

    def mate_parts(part1, part2):
        parent1 = part1.copy()

        min_mating_amount = len(part1) // 4

        two_point_order_crossover(part1, part2, min_mating_amount, rand)
        two_point_order_crossover(part2, parent1, min_mating_amount, rand)

    order1, order2 = child1[0], child2[0]

    # mate parts inside priority groups

    # priorities of tasks with same order-index should be the same (if chromosome is valid)
    cur_priority = priorities[order1[0]]
    cur_priority_group_start = 0
    for i in range(len(order1)):
        if priorities[order1[i]] != cur_priority:
            cur_priority = priorities[order1[i]]

            mate_parts(order1[cur_priority_group_start:i], order2[cur_priority_group_start:i])

    return toolbox.Individual(child1), toolbox.Individual(child2)


def two_point_order_crossover(child: np.ndarray, other_parent: np.ndarray, min_mating_amount: int, rand: random.Random):
    """Perform two-point crossover on order chromosome.

    Выполняет двухточечный кроссовер на хромосоме порядка.

    Args:
        child (np.ndarray): Order of first parent.
            Порядок первого родителя.
        other_parent (np.ndarray): Order of second parent.
            Порядок второго родителя.
        min_mating_amount (int): Minimum crossover segment.
            Минимальный сегмент кроссовера.
        rand (random.Random): Random generator.
            Генератор случайных чисел.

    Returns:
        np.ndarray: Updated order.
            Обновленный порядок.
    """
    # randomly select mating amount for child
    mating_amount = rand.randint(min_mating_amount, 3 * min_mating_amount)
    if mating_amount > 1:
        # based on received mating amount randomly select the points between which the crossover will take place
        crossover_head_point = rand.randint(1, mating_amount - 1)
        crossover_tail_point = mating_amount - crossover_head_point

        # get mating order part from parent
        ind_new_part = get_order_part(np.concatenate((child[:crossover_head_point], child[-crossover_tail_point:])),
                                      other_parent)
        # update mating part to received values
        child[crossover_head_point:-crossover_tail_point] = ind_new_part

    return child


def mutate_scheduling_order_core(order: np.ndarray, mutpb: float, rand: random.Random,
                                 parents: dict[int, set[int]], children: dict[int, set[int]]):
    """Core mutation for work order respecting dependencies.

    Ядро мутации порядка работ с учетом зависимостей.

    Args:
        order (np.ndarray): Current order.
            Текущий порядок.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        parents (dict[int, set[int]]): Parent mapping.
            Отображение родителей.
        children (dict[int, set[int]]): Children mapping.
            Отображение потомков.
    """
    # number of possible mutations = number of works except start and finish works
    num_possible_muts = len(order) - 2
    # generate mask of works to mutate based on mutation probability
    mask = np.array([rand.random() < mutpb for _ in range(num_possible_muts)])
    if mask.any():
        # get indexes of works to mutate based on generated mask
        # +1 because start work was not taken into account in mask generation
        indexes_of_works_to_mutate = np.where(mask)[0] + 1
        # shuffle order of mutations
        rand.shuffle(indexes_of_works_to_mutate)
        # get works to mutate based on shuffled indexes
        works_to_mutate = order[indexes_of_works_to_mutate]
        for work in works_to_mutate:
            # pop index of the current work
            i, indexes_of_works_to_mutate = indexes_of_works_to_mutate[0], indexes_of_works_to_mutate[1:]
            # find max index of parent of the current work
            # +1 because insertion should be righter
            i_parent = np.max(np.where(np.isin(order[:i], list(parents[work]), assume_unique=True))[0],
                              initial=0) + 1
            # find min index of child of the current work
            # +i because the slice [i + 1:] was taken, and +1 is not needed because these indexes will be shifted left
            # after current work deletion
            i_children = np.min(np.where(np.isin(order[i + 1:], list(children[work]), assume_unique=True))[0],
                                initial=len(order) - 2 - i) + i
            if i_parent == i_children:
                # if child and parent indexes are equal then no mutation can be done
                continue
            else:
                # shift work indexes (which are to the right of the current index) to the left
                # after the current work deletion
                indexes_of_works_to_mutate[indexes_of_works_to_mutate > i] -= 1
                # range potential indexes to insert the current work
                choices = np.concatenate((np.arange(i_parent, i), np.arange(i + 1, i_children + 1)))
                # set weights to potential indexes based on their distance from the current one
                weights = 1 / np.abs(choices - i)
                # generate new index for the current work
                new_i = rand.choices(choices, weights=weights)[0]
                # delete current work from current index, insert in new generated index and update scheduling order
                # in chromosome
                order[:] = np.insert(np.delete(order, i), new_i, work)
                # shift work indexes (which are to the right or equal to the new index) to the right
                # after the current work insertion in new generated index
                indexes_of_works_to_mutate[indexes_of_works_to_mutate >= new_i] += 1


def mutate_scheduling_order(ind: Individual, mutpb: float, rand: random.Random, priorities: np.ndarray,
                            parents: dict[int, set[int]], children: dict[int, set[int]]) -> Individual:
    """Mutate work order of an individual.

    Мутирует порядок работ индивида.

    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.
        parents (dict[int, set[int]]): Parent mapping for order validity.
            Отображение родителей для валидности порядка.
        children (dict[int, set[int]]): Children mapping for order validity.
            Отображение потомков для валидности порядка.

    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    """
    order = ind[0]

    priority_groups_count = len(set(priorities))
    mutpb_for_priority_group = mutpb #/ priority_groups_count

    # priorities of tasks with same order-index should be the same (if chromosome is valid)
    cur_priority = priorities[order[0]]
    cur_priority_group_start = 0
    for i in range(len(order)):
        if priorities[order[i]] != cur_priority:
            mutate_scheduling_order_core(order[cur_priority_group_start:i],
                                         mutpb_for_priority_group,
                                         rand, parents, children)

            cur_priority = priorities[order[i]]
            cur_priority_group_start = i

    return ind


def mate_resources(ind1: Individual, ind2: Individual, rand: random.Random,
                   optimize_resources: bool, toolbox: Toolbox, copy: bool = True) -> tuple[Individual, Individual]:
    """One-point crossover for resources.

    Одноточечный кроссовер ресурсов.

    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        optimize_resources (bool): Update resource borders after mating.
            Обновлять ли границы ресурсов после скрещивания.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        copy (bool): Copy individuals before mating.
            Копировать ли индивидов перед скрещиванием.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.

    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    """
    child1, child2 = (toolbox.copy_individual(ind1), toolbox.copy_individual(ind2)) if copy else (ind1, ind2)

    res1, res2 = child1[1], child2[1]
    num_works = len(res1)
    min_mating_amount = num_works // 4
    cxpoint = rand.randint(min_mating_amount, num_works - min_mating_amount)
    mate_positions = rand.sample(range(num_works), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]

    if optimize_resources:
        for res, child in zip([res1, res2], [child1, child2]):
            mated_resources = res[mate_positions]
            # take contractors from mated positions
            contractors = np.unique(mated_resources[:, -1])
            # take maximum from borders of these contractors in two chromosomes to maintain validity
            # and update current child borders on received maximum
            child[2][contractors] = np.stack((child1[2][contractors], child2[2][contractors]), axis=0).max(axis=0)

    return toolbox.Individual(child1), toolbox.Individual(child2)


def mutate_resources(ind: Individual, mutpb: float, rand: random.Random,
                     resources_border: np.ndarray) -> Individual:
    """Mutate resources of an individual.

    Мутирует ресурсы индивида.

    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        resources_border (np.ndarray): Lower and upper resource borders.
            Нижние и верхние границы ресурсов.

    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    """
    res = ind[1]
    num_works = len(res)

    num_contractors = len(ind[2])
    if num_contractors > 1:
        mask = np.array([rand.random() < mutpb for _ in range(num_works)])
        if mask.any():
            # generate new contractors in the number of received True values of mask
            new_contractors = np.array([rand.randint(0, num_contractors - 1) for _ in range(mask.sum())])
            # obtain a new mask of correspondence
            # between the borders of the received contractors and the assigned resources
            contractor_mask = (res[mask, :-1] <= ind[2][new_contractors]).all(axis=1)
            # update contractors by received mask
            new_contractors = new_contractors[contractor_mask]
            # update mask by new mask
            mask[mask] &= contractor_mask
            # mutate contractors
            res[mask, -1] = new_contractors

    num_res = len(res[0, :-1])
    res_indexes = np.arange(0, num_res)
    works_indexes = np.arange(0, num_works)
    masks = np.array([[rand.random() < mutpb for _ in range(num_res)] for _ in range(num_works)])
    # mask of works where at least one resource should be mutated
    mask = masks.any(axis=1)

    if not mask.any():
        # if no True value in mask then no mutation can be done
        return ind

    # get works indexes where mutation should be done and their masks of resources to be mutated
    works_indexes, masks = works_indexes[mask], masks[mask]
    # get up borders of resources of works where mutation should be done
    # by taking minimum (borders of the contractors assigned to them) and (maximum values of resources for these works)
    res_up_borders = np.stack((resources_border[1].T[mask], ind[2][res[mask, -1]]), axis=0).min(axis=0)
    # get minimum values of resources for these works
    res_low_borders = resources_border[0].T[mask]
    # if low border and up border are equal then no mutation can be done
    # update masks by checking this condition
    masks &= res_up_borders != res_low_borders
    # update mask of works where mutation should be done
    mask = masks.any(axis=1)

    # make mutation of resources
    mutate_values(res, works_indexes[mask], res_indexes, res_low_borders[mask],
                  res_up_borders[mask], masks[mask], -1, rand)

    return ind


def mate(ind1: Individual, ind2: Individual, optimize_resources: bool,
         rand: random.Random, toolbox: Toolbox, priorities: np.ndarray) \
        -> tuple[Individual, Individual]:
    """Combined crossover for order, resources, and zones.

    Комбинированный кроссовер порядка, ресурсов и зон.

    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        optimize_resources (bool): Adjust borders after mating.
            Изменять границы после скрещивания.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.

    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    """
    child1, child2 = mate_scheduling_order(ind1, ind2, rand, toolbox, priorities, copy=True)
    child1, child2 = mate_resources(child1, child2, rand, optimize_resources, toolbox, copy=False)
    # TODO Make better crossover for zones and uncomment this
    # child1, child2 = mate_for_zones(child1, child2, rand, copy=False)

    return toolbox.Individual(child1), toolbox.Individual(child2)


def mutate(ind: Individual, resources_border: np.ndarray, parents: dict[int, set[int]],
           children: dict[int, set[int]], statuses_available: int, priorities: np.ndarray,
           order_mutpb: float, res_mutpb: float, zone_mutpb: float,
           rand: random.Random) -> Individual:
    """Combined mutation for order, resources, and zones.

    Комбинированная мутация порядка, ресурсов и зон.

    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        resources_border (np.ndarray): Resource borders.
            Границы ресурсов.
        parents (dict[int, set[int]]): Parents mapping.
            Отображение родителей.
        children (dict[int, set[int]]): Children mapping.
            Отображение потомков.
        statuses_available (int): Number of statuses.
            Количество статусов.
        order_mutpb (float): Order mutation probability.
            Вероятность мутации порядка.
        res_mutpb (float): Resource mutation probability.
            Вероятность мутации ресурсов.
        zone_mutpb (float): Zone mutation probability.
            Вероятность мутации зон.
        rand (random.Random): Random generator.
            Генератор случайных чисел.

    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    """
    mutant = mutate_scheduling_order(ind, order_mutpb, rand, priorities, parents, children)
    mutant = mutate_resources(mutant, res_mutpb, rand, resources_border)
    # TODO Make better mutation for zones and uncomment this
    # mutant = mutate_for_zones(mutant, statuses_available, zone_mutpb, rand)

    return mutant


def mutate_resource_borders(ind: Individual, mutpb: float, rand: random.Random,
                            contractor_borders: np.ndarray) -> Individual:
    """Mutate contractors' resource borders.

    Мутирует границы ресурсов подрядчиков.

    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        contractor_borders (np.ndarray): Upper capacity borders.
            Верхние границы мощностей.

    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    """
    borders = ind[2]
    res = ind[1]
    num_res = len(res[0, :-1])
    res_indexes = np.arange(0, num_res)
    # sort resource part of chromosome by contractor ids
    resources = res[res[:, -1].argsort()]
    # get unique contractors and indexes where they start
    contractors, indexes = np.unique(resources[:, -1], return_index=True)
    # split resources to get parts grouped by contractor parts
    res_grouped_by_contractor = np.split(resources[:, :-1], indexes[1:])
    masks = np.array([[rand.random() < mutpb for _ in range(num_res)] for _ in contractors])
    # mask of contractors where at least one resource border should be mutated
    mask = masks.any(axis=1)

    if not mask.any():
        # if no True value in mask then no mutation can be done
        return ind

    # get contractors where mutation should be done and their masks of resource borders to be mutated
    contractors, masks = contractors[mask], masks[mask]
    # get maximum values of resource borders for received contractors
    contractor_up_borders = contractor_borders[contractors]
    # get minimum values of resource borders of contractors where mutation should be done
    # by taking maximum of assigned resources for works which have contractor that should be mutated
    contractor_low_borders = np.array([r.max(axis=0) for r, is_mut in zip(res_grouped_by_contractor, mask) if is_mut])
    # if minimum and maximum values are equal then no mutation can be done
    # update masks by checking this condition
    masks &= contractor_up_borders != contractor_low_borders
    # update mask of contractors where mutation should be done
    mask = masks.any(axis=1)

    # make mutation of resource borders
    mutate_values(borders, contractors[mask], res_indexes,
                  contractor_low_borders[mask], contractor_up_borders[mask],
                  masks[mask], len(res_indexes), rand)

    return ind


def mutate_values(chromosome_part: np.ndarray, row_indexes: np.ndarray, col_indexes: np.ndarray,
                  low_borders: np.ndarray, up_borders: np.ndarray, masks: np.ndarray, mut_part: int,
                  rand: random.Random) -> None:
    """Change numeric values in chromosome slice.

    Изменяет числовые значения в части хромосомы.
    """
    for row_index, l_borders, u_borders, row_mask in zip(row_indexes, low_borders, up_borders, masks):
        cur_row = chromosome_part[row_index]
        for col_index, current_amount, l_border, u_border in zip(col_indexes[row_mask], cur_row[:mut_part][row_mask],
                                                                 l_borders[row_mask], u_borders[row_mask]):
            # range new potential amount except current amount
            choices = np.concatenate((np.arange(l_border, current_amount),
                                      np.arange(current_amount + 1, u_border + 1)))
            # set weights to potential amounts based on their distance from the current one
            weights = 1 / np.abs(choices - current_amount)
            cur_row[col_index] = rand.choices(choices, weights=weights)[0]


def mate_for_zones(ind1: Individual, ind2: Individual, rand: random.Random,
                   toolbox: Toolbox, copy: bool = True) -> tuple[Individual, Individual]:
    """One-point crossover for zones.

    Одноточечный кроссовер зон.

    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        copy (bool): Copy individuals before mating.
            Копировать ли индивидов перед скрещиванием.

    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    """
    child1, child2 = (toolbox.copy_individual(ind1), toolbox.copy_individual(ind2)) if copy else (ind1, ind2)

    zones1 = child1[4]
    zones2 = child2[4]

    if zones1.size:
        num_works = len(zones1)
        border = num_works // 4
        cxpoint = rand.randint(border, num_works - border)

        mate_positions = rand.sample(range(num_works), cxpoint)

        zones1[mate_positions], zones2[mate_positions] = zones2[mate_positions], zones1[mate_positions]

    return toolbox.Individual(child1), toolbox.Individual(child2)


def mutate_for_zones(ind: Individual, mutpb: float, rand: random.Random, statuses_available: int) -> Individual:
    """Mutate zone statuses of an individual.

    Мутирует статус зон у индивида.

    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        statuses_available (int): Number of available statuses.
            Количество доступных статусов.

    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    """
    # select random number from interval from min to max from uniform distribution
    zones = ind[4]
    if zones.size:
        mask = np.array([[rand.random() < mutpb for _ in range(zones.shape[1])] for _ in range(zones.shape[0])])
        new_zones = np.array([rand.randint(0, statuses_available - 1) for _ in range(mask.sum())])
        zones[mask] = new_zones

    return ind
