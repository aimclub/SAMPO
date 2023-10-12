import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from operator import attrgetter
from typing import Iterable

import numpy as np
from deap import creator, base
from sortedcontainers import SortedList

from sampo.scheduler.genetic.converter import convert_chromosome_to_schedule
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ChromosomeType
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler
from sampo.scheduler.utils.peaks import get_absolute_peak_resource_usage
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.resource_cost import schedule_cost


# logger = mp.log_to_stderr(logging.DEBUG)


class FitnessFunction(ABC):
    """
    Base class for description of different fitness functions.
    """

    def __init__(self, deadline: Time | None):
        self._deadline = deadline

    @abstractmethod
    def evaluate(self, schedules: list[Schedule]) -> list[int]:
        """
        Calculate the value of fitness function of the all schedules.
        It is better when value is less.
        """
        ...


class TimeFitness(FitnessFunction):
    """
    Fitness function that relies on finish time.
    """

    def __init__(self, deadline: Time | None = None):
        super().__init__(deadline)

    def evaluate(self, schedules: list[Schedule]) -> list[int]:
        return [schedule.execution_time.value for schedule in schedules]


class TimeAndResourcesFitness(FitnessFunction):
    """
    Fitness function that relies on finish time and the set of resources.
    """

    def __init__(self, deadline: Time | None = None):
        super().__init__(deadline)

    def evaluate(self, schedules: list[Schedule]) -> list[int]:
        return [schedule.execution_time.value + get_absolute_peak_resource_usage(schedule) for schedule in schedules]


class DeadlineResourcesFitness(FitnessFunction):
    """
    The fitness function is dependent on the set of resources and requires the end time to meet the deadline.
    """

    def __init__(self, deadline: Time):
        super().__init__(deadline)

    def evaluate(self, schedules: list[Schedule]) -> list[int]:
        return [int(get_absolute_peak_resource_usage(schedule)
                    * max(1.0, schedule.execution_time.value / self._deadline.value))
                for schedule in schedules]


class DeadlineCostFitness(FitnessFunction):
    """
    The fitness function is dependent on the cost of resources and requires the end time to meet the deadline.
    """

    def __init__(self, deadline: Time):
        super().__init__(deadline)

    def evaluate(self, schedules: list[Schedule]) -> list[int]:
        # TODO Integrate cost calculation to native module
        return [int(schedule_cost(schedule) * max(1.0, schedule.execution_time.value / self._deadline.value))
                for schedule in schedules]


# create class FitnessMin, the weights = -1 means that fitness - is function for minimum

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)
Individual = creator.Individual


class IndividualType(Enum):
    """
    Class to define a type of individual in genetic algorithm
    """
    Population = 'population'
    Offspring = 'offspring'


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
                 population_size: int,
                 rand: random.Random,
                 spec: ScheduleSpec,
                 worker_pool_indices: dict[int, dict[int, Worker]],
                 contractor2index: dict[str, int],
                 contractor_borders: np.ndarray,
                 node_indices: list[int],
                 parents: dict[int, set[int]],
                 resources_border: np.ndarray,
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> base.Toolbox:
    """
    Object, that include set of functions (tools) for genetic model and other functions related to it.
    list of parameters that received this function is sufficient and complete to manipulate with genetic algorithm

    :return: Object, included tools for genetic algorithm
    """
    toolbox = base.Toolbox()
    # generate chromosome
    toolbox.register('generate_chromosome', generate_chromosome, wg=wg, contractors=contractors,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape)

    # create population
    # toolbox.register('population', tools.initRepeat, list, lambda: toolbox.generate_chromosome())
    toolbox.register('population', generate_population, wg=wg, contractors=contractors,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape)
    # selection
    toolbox.register('select', select_new_population, pop_size=population_size)
    # combined crossover
    toolbox.register('mate', mate, rand=rand)
    # combined mutation
    toolbox.register('mutate', mutate, order_mutpb=mut_order_pb, res_mutpb=mut_res_pb, zone_mutpb=mut_zone_pb,
                     rand=rand, parents=parents, resources_border=resources_border, statuses_available=statuses_available)
    # crossover for order
    toolbox.register('mate_order', mate_scheduling_order, rand=rand)
    # mutation for order
    toolbox.register('mutate_order', mutate_scheduling_order, mutpb=mut_order_pb, rand=rand, parents=parents)
    # crossover for resources
    toolbox.register('mate_resources', mate_resources, rand=rand)
    # mutation for resources
    toolbox.register('mutate_resources', mutate_resources, resources_border=resources_border,
                     mutpb=mut_res_pb, rand=rand)
    # mutation for resource borders
    toolbox.register('mutate_resource_borders', mutate_resource_borders, contractor_borders=contractor_borders,
                     mutpb=mut_res_pb, rand=rand)
    toolbox.register('mate_post_zones', mate_for_zones, rand=rand)
    toolbox.register('mutate_post_zones', mutate_for_zones, rand=rand, mutpb=mut_zone_pb,
                     statuses_available=landscape.zone_config.statuses.statuses_available())

    toolbox.register('validate', is_chromosome_correct, node_indices=node_indices, parents=parents,
                     contractor_borders=contractor_borders)
    toolbox.register('schedule_to_chromosome', convert_schedule_to_chromosome, wg=wg,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     landscape=landscape)
    toolbox.register("chromosome_to_schedule", convert_chromosome_to_schedule, worker_pool=worker_pool,
                     index2node=index2node, index2contractor=index2contractor_obj,
                     worker_pool_indices=worker_pool_indices, assigned_parent_time=assigned_parent_time,
                     work_estimator=work_estimator, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, index2zone=index2zone,
                     landscape=landscape)
    toolbox.register('copy_individual', lambda ind: Individual(copy_chromosome(ind)))
    toolbox.register('update_resource_borders_to_peak_values', update_resource_borders_to_peak_values,
                     worker_name2index=worker_name2index, contractor2index=contractor2index)
    return toolbox


def copy_chromosome(chromosome: ChromosomeType) -> ChromosomeType:
    return chromosome[0].copy(), chromosome[1].copy(), chromosome[2].copy(), \
        deepcopy(chromosome[3]), chromosome[4].copy()


def generate_population(n: int,
                        wg: WorkGraph,
                        contractors: list[Contractor],
                        spec: ScheduleSpec,
                        work_id2index: dict[str, int],
                        worker_name2index: dict[str, int],
                        contractor2index: dict[str, int],
                        contractor_borders: np.ndarray,
                        init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                        rand: random.Random,
                        work_estimator: WorkTimeEstimator = None,
                        landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[Individual]:
    """
    Generates population.
    Do not use `generate_chromosome` function.
    """

    def randomized_init() -> ChromosomeType:
        schedule = RandomizedTopologicalScheduler(work_estimator, int(rand.random() * 1000000)) \
            .schedule(wg, contractors, landscape=landscape)
        return convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, spec, landscape)

    count_for_specified_types = (n // 3) // len(init_chromosomes)
    count_for_specified_types = count_for_specified_types if count_for_specified_types > 0 else 1
    sum_counts_for_specified_types = count_for_specified_types * len(init_chromosomes)
    counts = [count_for_specified_types * importance for _, importance, _ in init_chromosomes.values()]

    weights_multiplier = math.ceil(sum_counts_for_specified_types / sum(counts))
    counts = [count * weights_multiplier for count in counts]

    count_for_topological = n - sum_counts_for_specified_types
    count_for_topological = count_for_topological if count_for_topological > 0 else 1
    counts += [count_for_topological]

    chromosome_types = rand.sample(list(init_chromosomes.keys()) + ['topological'], k=n, counts=counts)

    chromosomes = [Individual(init_chromosomes[generated_type][0])
                   if generated_type != 'topological' else Individual(randomized_init())
                   for generated_type in chromosome_types]

    return chromosomes


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
                        landscape: LandscapeConfiguration = LandscapeConfiguration()) -> Individual:
    """
    It is necessary to generate valid scheduling, which are satisfied to current dependencies
    That's why will be used the approved order of works (HEFT order and Topological sorting)
    Topological sorts are generating always different
    HEFT is always the same(now not)
    HEFT we will choose in 30% of attempts
    Topological in others
    """

    def randomized_init() -> ChromosomeType:
        schedule = RandomizedTopologicalScheduler(work_estimator,
                                                  int(rand.random() * 1000000)) \
            .schedule(wg, contractors, spec, landscape=landscape)
        return convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, spec, landscape)

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

    return Individual(chromosome)


def select_new_population(population: list[ChromosomeType], pop_size: int) -> list[ChromosomeType]:
    """
    Selection operator for genetic algorithm.
    Select top n individuals in population.
    """
    population = sorted(population, key=attrgetter('fitness'), reverse=True)
    return population[:pop_size]


def is_chromosome_correct(chromosome: ChromosomeType, node_indices: list[int], parents: dict[int, set[int]],
                          contractor_borders: np.ndarray) -> bool:
    """
    Check correctness of works order and contractors borders.
    """
    return is_chromosome_order_correct(chromosome, parents) and \
        is_chromosome_contractors_correct(chromosome, node_indices, contractor_borders)


def is_chromosome_order_correct(chromosome: ChromosomeType, parents: dict[int, set[int]]) -> bool:
    """
    Checks that assigned order of works are topologically correct.
    """
    work_order = chromosome[0]
    used = set()
    for work_index in work_order:
        used.add(work_index)
        if not parents[work_index].issubset(used):
            # logger.error(f'Order validation failed: {work_order}')
            return False
    return True


def is_chromosome_contractors_correct(chromosome: ChromosomeType, work_indices: Iterable[int],
                                      contractor_borders: np.ndarray) -> bool:
    """
    Checks that assigned contractors can supply assigned workers.
    """
    if not work_indices:
        return True
    resources = chromosome[1][work_indices]
    resources = resources[resources[:, -1].argsort()]
    contractors, indexes = np.unique(resources[:, -1], return_index=True)
    chromosome_borders = chromosome[2][contractors]
    res_grouped_by_contractor = np.split(resources[:, :-1], indexes[1:])
    max_of_res_by_contractor = np.array([r.max(axis=0) for r in res_grouped_by_contractor])
    return (max_of_res_by_contractor <= chromosome_borders).all() and \
        (chromosome_borders <= contractor_borders[contractors]).all()


def get_order_part(order: np.ndarray, other_order: np.ndarray) -> np.ndarray:
    """
    Get a new part in topologic order for chromosome.
    This function is needed to make crossover for order.
    """
    order = set(order)
    return np.array([node for node in other_order if node not in order])


def mate_scheduling_order(ind1: ChromosomeType, ind2: ChromosomeType, rand: random.Random, copy: bool = True) \
        -> tuple[ChromosomeType, ChromosomeType]:
    """
    Two-Point crossover for order.

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for randomized operations
    :param copy: if True individuals will be copied before mating so as not to change them

    :return: two mated individuals
    """
    child1, child2 = (Individual(copy_chromosome(ind1)), Individual(copy_chromosome(ind2))) if copy else (ind1, ind2)

    order1, order2 = child1[0], child2[0]

    min_mating_amount = len(order1) // 4

    # randomly select the points where the crossover will take place
    mating_amount = rand.randint(min_mating_amount, 3 * min_mating_amount)
    if mating_amount > 1:
        crossover_head_point = rand.randint(1, mating_amount - 1)
        crossover_tail_point = mating_amount - crossover_head_point

        ind_new_part = get_order_part(np.concatenate((order1[:crossover_head_point], order1[-crossover_tail_point:])),
                                      order2)
        order1[crossover_head_point:-crossover_tail_point] = ind_new_part

    # randomly select the points where the crossover will take place
    mating_amount = rand.randint(min_mating_amount, 3 * min_mating_amount)
    if mating_amount > 1:
        crossover_head_point = rand.randint(1, mating_amount - 1)
        crossover_tail_point = mating_amount - crossover_head_point

        ind_new_part = get_order_part(np.concatenate((order2[:crossover_head_point], order2[-crossover_tail_point:])),
                                      order1)
        order2[crossover_head_point:-crossover_tail_point] = ind_new_part

    return child1, child2


def mutate_scheduling_order(ind: ChromosomeType, mutpb: float, rand: random.Random,
                            parents: dict[int, set[int]]) -> ChromosomeType:
    """
    Mutation operator for order.
    Swap neighbors.

    :param ind: the individual to be mutated
    :param mutpb: probability of gene mutation
    :param rand: the rand object used for randomized operations
    :param parents: mapping object of works and their parent-works to create valid order

    :return: mutated individual
    """
    order = ind[0]
    num_possible_muts = len(order) - 3
    mask = np.array([rand.random() < mutpb for _ in range(num_possible_muts)])
    if mask.any():
        indexes_to_mutate = [rand.randint(1, num_possible_muts + 1) for _ in range(mask.sum())]
        for i in indexes_to_mutate:
            if order[i] not in parents[order[i + 1]]:
                order[i], order[i + 1] = order[i + 1], order[i]

    return ind


def mate_resources(ind1: ChromosomeType, ind2: ChromosomeType, optimize_resources: bool,
                   rand: random.Random, copy: bool = True) -> tuple[ChromosomeType, ChromosomeType]:
    """
    One-Point crossover for resources.

    :param ind1: first individual
    :param ind2: second individual
    :param optimize_resources: if True resource borders should be changed after mating
    :param rand: the rand object used for randomized operations
    :param copy: if True individuals will be copied before mating so as not to change them

    :return: two mated individuals
    """
    child1, child2 = (Individual(copy_chromosome(ind1)), Individual(copy_chromosome(ind2))) if copy else (ind1, ind2)

    res1, res2 = child1[1], child2[1]
    num_works = len(res1)
    min_mating_amount = num_works // 4
    cxpoint = rand.randint(min_mating_amount, num_works - min_mating_amount)
    mate_positions = rand.sample(range(num_works), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]

    if optimize_resources:
        for res, child in zip([res1, res2], [child1, child2]):
            mated_resources = res[mate_positions]
            contractors = np.unique(mated_resources[:, -1])
            child[2][contractors] = np.stack((child1[2][contractors], child2[2][contractors]), axis=0).max(axis=0)

    return child1, child2


def mutate_resources(ind: ChromosomeType, mutpb: float, rand: random.Random,
                     resources_border: np.ndarray) -> ChromosomeType:
    """
    Mutation function for resources.
    It changes selected numbers of workers in random work in a certain interval for this work.

    :param ind: the individual to be mutated
    :param resources_border: low and up borders of resources amounts
    :param mutpb: probability of gene mutation
    :param rand: the rand object used for randomized operations

    :return: mutated individual
    """
    res = ind[1]
    num_works = len(res)

    num_contractors = len(ind[2])
    if num_contractors > 1:
        mask = np.array([rand.random() < mutpb for _ in range(num_works)])
        if mask.any():
            new_contractors = np.array([rand.randint(0, num_contractors - 1) for _ in range(mask.sum())])
            contractor_mask = (res[mask, :-1] <= ind[2][new_contractors]).all(axis=1)
            new_contractors = new_contractors[contractor_mask]
            mask[mask] &= contractor_mask
            res[mask, -1] = new_contractors

    num_res = len(res[0, :-1])
    res_indexes = np.arange(0, num_res)
    works_indexes = np.arange(0, num_works)
    masks = np.array([[rand.random() < mutpb for _ in range(num_res)] for _ in range(num_works)])
    mask = masks.any(axis=1)

    if not mask.any():
        return ind

    works_indexes, masks = works_indexes[mask], masks[mask]
    res_up_borders = np.stack((resources_border[1].T[mask], ind[2][res[mask, -1]]), axis=0).min(axis=0)
    res_low_borders = resources_border[0].T[mask]
    masks &= res_up_borders != res_low_borders
    mask = masks.any(axis=1)

    mutate_values(res, works_indexes[mask], res_indexes, res_low_borders[mask], res_up_borders[mask], masks[mask], -1,
                  rand)

    return ind


def mate(ind1: ChromosomeType, ind2: ChromosomeType, optimize_resources: bool, rand: random.Random) \
        -> tuple[ChromosomeType, ChromosomeType]:
    """
    Combined crossover function of Two-Point crossover for order and One-Point crossover for resources.

    :param ind1: first individual
    :param ind2: second individual
    :param optimize_resources: if True resource borders should be changed after mating
    :param rand: the rand object used for randomized operations

    :return: two mated individuals
    """
    child1, child2 = mate_scheduling_order(ind1, ind2, rand, copy=True)
    child1, child2 = mate_resources(child1, child2, optimize_resources, rand, copy=False)
    # child1, child2 = mate_for_zones(child1, child2, rand, copy=False)

    return child1, child2


def mutate(ind: ChromosomeType, resources_border: np.ndarray, parents: dict[int, set[int]],
           order_mutpb: float, res_mutpb: float, zone_mutpb: float, statuses_available: int,
           rand: random.Random) -> ChromosomeType:
    """
    Combined mutation function of mutation for order and mutation for resources.

    :param ind: the individual to be mutated
    :param resources_border: low and up borders of resources amounts
    :param parents: mapping object of works and their parent-works to create valid order
    :param order_mutpb: probability of order's gene mutation
    :param res_mutpb: probability of resources' gene mutation
    :param rand: the rand object used for randomized operations

    :return: mutated individual
    """
    mutant = mutate_scheduling_order(ind, order_mutpb, rand, parents)
    mutant = mutate_resources(mutant, res_mutpb, rand, resources_border)
    # mutant = mutate_for_zones(mutant, statuses_available, zone_mutpb, rand)

    return mutant


def mutate_resource_borders(ind: ChromosomeType, mutpb: float, rand: random.Random,
                            contractor_borders: np.ndarray) -> ChromosomeType:
    """
    Mutation for contractors' resource borders.

    :param ind: the individual to be mutated
    :param contractor_borders: up borders of contractors capacity
    :param mutpb: probability of gene mutation
    :param rand: the rand object used for randomized operations

    :return: mutated individual
    """
    borders = ind[2]
    res = ind[1]
    num_res = len(res[0, :-1])
    res_indexes = np.arange(0, num_res)
    resources = res[res[:, -1].argsort()]
    contractors, indexes = np.unique(resources[:, -1], return_index=True)
    res_grouped_by_contractor = np.split(resources[:, :-1], indexes[1:])
    masks = np.array([[rand.random() < mutpb for _ in range(num_res)] for _ in contractors])
    mask = masks.any(axis=1)

    if not mask.any():
        return ind

    contractors, masks = contractors[mask], masks[mask]
    contractor_up_borders = contractor_borders[contractors]
    contractor_low_borders = np.array([r.max(axis=0) for r, is_mut in zip(res_grouped_by_contractor, mask) if is_mut])
    masks &= contractor_up_borders != contractor_low_borders
    mask = masks.any(axis=1)

    mutate_values(borders, contractors[mask], res_indexes, contractor_low_borders[mask], contractor_up_borders[mask],
                  masks[mask], len(res_indexes), rand)

    return ind


def mutate_values(chromosome_part: np.ndarray, row_indexes: np.ndarray, col_indexes: np.ndarray,
                  low_borders: np.ndarray, up_borders: np.ndarray, masks: np.ndarray, mut_part: int,
                  rand: random.Random) -> None:
    """
    Changes numeric values in m x n part of chromosome.
    This function is needed to make mutation for resources and resource borders.
    """
    for row_index, l_borders, u_borders, row_mask in zip(row_indexes, low_borders, up_borders, masks):
        cur_row = chromosome_part[row_index]
        for col_index, current_amount, l_border, u_border in zip(col_indexes[row_mask], cur_row[:mut_part][row_mask],
                                                                 l_borders[row_mask], u_borders[row_mask]):
            choices = np.concatenate((np.arange(l_border, current_amount),
                                      np.arange(current_amount + 1, u_border + 1)))
            weights = 1 / abs(choices - current_amount)
            cur_row[col_index] = rand.choices(choices, weights=weights)[0]


def update_resource_borders_to_peak_values(ind: ChromosomeType, schedule: Schedule, worker_name2index: dict[str, int],
                                           contractor2index: dict[str, int]):
    """
    Changes the resource borders to the peak values obtained in the schedule.

    :param ind: the individual to be updated
    :param schedule: schedule obtained from the individual
    :param worker_name2index: mapping object of resources and their index in chromosome
    :param contractor2index: mapping object of contractors and their index in chromosome

    :return: individual with updated resource borders
    """
    df = schedule.full_schedule_df
    contractors = set(df.contractor)
    actual_borders = np.zeros_like(ind[2])
    for contractor in contractors:
        contractor_df = df[df.contractor == contractor]
        points = contractor_df[['start', 'finish']].to_numpy().copy()
        points[:, 1] += 1
        points = SortedList(set(points.flatten()))
        contractor_res_schedule = np.zeros((len(points), len(worker_name2index)))
        contractor_id = ''
        for _, r in contractor_df.iterrows():
            start = points.bisect_left(r['start'])
            finish = points.bisect_left(r['finish'] + 1)
            swork = r['scheduled_work_object']
            workers = np.array([[worker_name2index[worker.name], worker.count] for worker in swork.workers])
            if len(workers):
                contractor_res_schedule[start: finish, workers[:, 0]] += workers[:, 1]
                if not contractor_id:
                    contractor_id = swork.workers[0].contractor_id
        if contractor_id:
            index = contractor2index[contractor_id]
            actual_borders[index] = contractor_res_schedule.max(axis=0)
    ind[2][:] = actual_borders
    return ind


def mate_for_zones(ind1: ChromosomeType, ind2: ChromosomeType,
                   rand: random.Random, copy: bool = True) -> tuple[ChromosomeType, ChromosomeType]:
    """
    CxOnePoint for zones

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for exchange point selection
    :return: first and second individual
    """
    child1, child2 = (Individual(copy_chromosome(ind1)), Individual(copy_chromosome(ind2))) if copy else (ind1, ind2)

    res1 = child1[4]
    res2 = child2[4]
    num_works = len(res1)
    border = num_works // 4
    cxpoint = rand.randint(border, num_works - border)

    mate_positions = rand.sample(range(num_works), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]
    return child1, child2


def mutate_for_zones(ind: ChromosomeType, statuses_available: int,
                     mutpb: float, rand: random.Random) -> ChromosomeType:
    """
    Mutation function for zones.
    It changes selected numbers of zones in random work in a certain interval from available statuses.

    :return: mutate individual
    """
    # select random number from interval from min to max from uniform distribution
    res = ind[4]
    for i, work_post_zones in enumerate(res):
        for type_of_zone in range(len(res[0])):
            if rand.random() < mutpb:
                work_post_zones[type_of_zone] = rand.randint(0, statuses_available - 1)

    return ind
