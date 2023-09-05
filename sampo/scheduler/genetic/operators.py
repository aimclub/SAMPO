import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from operator import attrgetter
from typing import Iterable, Callable

import numpy as np
from deap import creator, base

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

    def __init__(self, evaluator: Callable[[list[ChromosomeType]], list[Schedule]]):
        self._evaluator = evaluator

    @abstractmethod
    def evaluate(self, chromosomes: list[ChromosomeType]) -> list[int]:
        """
        Calculate the value of fitness function of the all chromosomes.
        It is better when value is less.
        """
        ...


class TimeFitness(FitnessFunction):
    """
    Fitness function that relies on finish time.
    """

    def __init__(self, evaluator: Callable[[list[ChromosomeType]], list[Schedule]]):
        super().__init__(evaluator)

    def evaluate(self, chromosomes: list[ChromosomeType]) -> list[int]:
        return [schedule.execution_time.value for schedule in self._evaluator(chromosomes)]


class TimeAndResourcesFitness(FitnessFunction):
    """
    Fitness function that relies on finish time and the set of resources.
    """

    def __init__(self, evaluator: Callable[[list[ChromosomeType]], list[Schedule]]):
        super().__init__(evaluator)

    def evaluate(self, chromosomes: list[ChromosomeType]) -> list[int]:
        evaluated = self._evaluator(chromosomes)
        return [schedule.execution_time.value + get_absolute_peak_resource_usage(schedule) for schedule in evaluated]


class DeadlineResourcesFitness(FitnessFunction):
    """
    The fitness function is dependent on the set of resources and requires the end time to meet the deadline.
    """

    def __init__(self, deadline: Time, evaluator: Callable[[list[ChromosomeType]], list[Schedule]]):
        super().__init__(evaluator)
        self._deadline = deadline

    @staticmethod
    def prepare(deadline: Time):
        """
        Returns the constructor of that fitness function prepared to use in Genetic
        """
        return partial(DeadlineResourcesFitness, deadline)

    def evaluate(self, chromosomes: list[ChromosomeType]) -> list[int]:
        evaluated = self._evaluator(chromosomes)
        return [int(get_absolute_peak_resource_usage(schedule)
                    * max(1.0, schedule.execution_time.value / self._deadline.value))
                for schedule in evaluated]


class DeadlineCostFitness(FitnessFunction):
    """
    The fitness function is dependent on the cost of resources and requires the end time to meet the deadline.
    """

    def __init__(self, deadline: Time, evaluator: Callable[[list[ChromosomeType]], list[Schedule]]):
        super().__init__(evaluator)
        self._deadline = deadline

    @staticmethod
    def prepare(deadline: Time):
        """
        Returns the constructor of that fitness function prepared to use in Genetic
        """
        return partial(DeadlineCostFitness, deadline)

    def evaluate(self, chromosomes: list[ChromosomeType]) -> list[int]:
        evaluated = self._evaluator(chromosomes)
        # TODO Integrate cost calculation to native module
        return [int(schedule_cost(schedule) * max(1.0, schedule.execution_time.value / self._deadline.value))
                for schedule in evaluated]


# create class FitnessMin, the weights = -1 means that fitness - is function for minimum

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)
Individual = creator.Individual


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
                 mutate_order: float,
                 mutate_resources: float,
                 mutate_zones: float,
                 population_size: int,
                 rand: random.Random,
                 spec: ScheduleSpec,
                 worker_pool_indices: dict[int, dict[int, Worker]],
                 contractor2index: dict[str, int],
                 contractor_borders: np.ndarray,
                 node_indices: list[int],
                 parents: dict[int, list[int]],
                 resources_border: np.ndarray,
                 resources_min_border: np.ndarray,
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> base.Toolbox:
    """
    Object, that include set of functions (tools) for genetic model and other functions related to it.
    list of parameters that received this function is sufficient and complete to manipulate with genetic

    :return: Object, included tools for genetic
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
    # crossover for order
    toolbox.register('mate', mate_scheduling_order, rand=rand)
    # mutation for order
    toolbox.register('mutate', mutate_scheduling_order, mutpb=mutate_order, rand=rand)
    # crossover for resources
    toolbox.register('mate_resources', mate_for_resources, rand=rand)
    # mutation for resources
    toolbox.register('mutate_resources', mutate_for_resources, resources_border=resources_border,
                     mutpb=mutate_resources, rand=rand)
    # crossover for resource borders
    toolbox.register('mate_resource_borders', mate_for_resource_borders, rand=rand)
    # mutation for resource borders
    toolbox.register('mutate_resource_borders', mutate_resource_borders, resources_min_border=resources_min_border,
                     mutpb=mutate_resources, rand=rand)
    toolbox.register('mate_post_zones', mate_for_zones, rand=rand)
    toolbox.register('mutate_post_zones', mutate_for_zones, rand=rand, mutpb=mutate_zones,
                     statuses_available=landscape.zone_config.statuses.statuses_available())

    toolbox.register('validate', is_chromosome_correct, node_indices=node_indices, parents=parents)
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
    Selecting top n individuals in population.
    """
    population = sorted(population, key=attrgetter('fitness'), reverse=True)
    return population[:pop_size]


def is_chromosome_correct(chromosome: ChromosomeType,
                          node_indices: list[int],
                          parents: dict[int, list[int]]) -> bool:
    """
    Check order of works and contractors.
    """
    return is_chromosome_order_correct(chromosome, parents) and \
        is_chromosome_contractors_correct(chromosome, node_indices)


def is_chromosome_order_correct(chromosome: ChromosomeType, parents: dict[int, list[int]]) -> bool:
    """
    Checks that assigned order of works are topologically correct.
    """
    work_order = chromosome[0]
    used = set()
    for work_index in work_order:
        used.add(work_index)
        for parent in parents[work_index]:
            if parent not in used:
                # logger.error(f'Order validation failed: {work_order}')
                return False
    return True


def is_chromosome_contractors_correct(chromosome: ChromosomeType,
                                      work_indices: Iterable[int]) -> bool:
    """
    Checks that assigned contractors can supply assigned workers.
    """
    for work_ind in work_indices:
        resources_count = chromosome[1][work_ind, :-1]
        contractor_ind = chromosome[1][work_ind, -1]
        contractor_border = chromosome[2][contractor_ind]
        for ind, count in enumerate(resources_count):
            if contractor_border[ind] < count:
                # logger.error(f'Contractor border validation failed: {contractor_border[ind]} < {count}')
                return False
    return True


def get_order_tail(head_set: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Get a new tail in topologic order for chromosome.
    This function is needed to make crossover for order.
    """
    head_set = set(head_set)
    return np.array([node for node in other if node not in head_set])


def mate_scheduling_order(ind1: ChromosomeType, ind2: ChromosomeType, rand: random.Random) \
        -> (ChromosomeType, ChromosomeType):
    """
    Crossover for order.
    Basis crossover is cxOnePoint.
    But we checked not repeated works in individual order.

    :return: two cross individuals
    """
    child1 = Individual(copy_chromosome(ind1))
    child2 = Individual(copy_chromosome(ind2))

    order1 = child1[0]
    order2 = child2[0]

    border = len(order1) // 4
    # randomly select the point where the crossover will take place
    crossover_point = rand.randint(border, len(order1) - border)

    ind1_new_tail = get_order_tail(order1[:crossover_point], order2)
    ind2_new_tail = get_order_tail(order2[:crossover_point], order1)

    order1[crossover_point:] = ind1_new_tail
    order2[crossover_point:] = ind2_new_tail

    return child1, child2


def mutate_scheduling_order(ind: ChromosomeType, mutpb: float, rand: random.Random) -> ChromosomeType:
    """
    Mutation operator for order.
    Swap neighbors.
    """
    order = ind[0]
    for i in range(1, len(order) - 2):
        if rand.random() < mutpb:
            order[i], order[i + 1] = order[i + 1], order[i]

    return ind


def mate_for_resources(ind1: ChromosomeType, ind2: ChromosomeType,
                       rand: random.Random) -> (ChromosomeType, ChromosomeType):
    """
    CxOnePoint for resources.

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for exchange point selection
    :return: first and second individual
    """
    child1 = Individual(copy_chromosome(ind1))
    child2 = Individual(copy_chromosome(ind2))

    res1 = child1[1]
    res2 = child2[1]
    num_works = len(res1)
    border = num_works // 4
    cxpoint = rand.randint(border, num_works - border)

    mate_positions = rand.sample(range(num_works), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]
    return child1, child2


def mutate_for_resources(ind: ChromosomeType, resources_border: np.ndarray,
                         mutpb: float, rand: random.Random) -> ChromosomeType:
    """
    Mutation function for resources.
    It changes selected numbers of workers in random work in a certain interval for this work.

    :return: mutate individual
    """
    # select random number from interval from min to max from uniform distribution
    res = ind[1]
    res_count = len(res[0])
    for i, work_res in enumerate(res):
        for type_of_res in range(res_count - 1):
            if rand.random() < mutpb:
                xl = resources_border[0, type_of_res, i]
                xu = resources_border[1, type_of_res, i]
                contractor = work_res[-1]
                border = ind[2][contractor, type_of_res]
                # TODO Debug why min(xu, border) can be lower than xl
                work_res[type_of_res] = rand.randint(xl, max(xl, min(xu, border)))
        if rand.random() < mutpb:
            work_res[-1] = rand.randint(0, len(ind[2]) - 1)

    return ind


def mate_for_resource_borders(ind1: ChromosomeType, ind2: ChromosomeType,
                              rand: random.Random) -> (ChromosomeType, ChromosomeType):
    """
    Crossover for contractors' resource borders.
    """
    child1 = Individual(copy_chromosome(ind1))
    child2 = Individual(copy_chromosome(ind2))

    borders1 = child1[2]
    borders2 = child2[2]
    num_contractors = len(borders1)
    contractors = rand.sample(range(num_contractors), rand.randint(1, num_contractors))

    num_res = len(borders1[0])
    res_indices = list(range(num_res))
    border = num_res // 4
    mate_positions = rand.sample(res_indices, rand.randint(border, num_res - border))

    (borders1[contractors, mate_positions],
     borders2[contractors, mate_positions]) = (borders2[contractors, mate_positions],
                                               borders1[contractors, mate_positions])

    return child1, child2


def mutate_resource_borders(ind: ChromosomeType, resources_min_border: np.ndarray,
                            mutpb: float, rand: random.Random) -> ChromosomeType:
    """
    Mutation for contractors' resource borders.
    """
    num_resources = len(resources_min_border)
    num_contractors = len(ind[2])
    type_of_res = rand.randint(0, len(ind[2][0]) - 1)
    for contractor in range(num_contractors):
        if rand.random() < mutpb:
            ind[2][contractor][type_of_res] -= rand.randint(resources_min_border[type_of_res] + 1,
                                                            max(resources_min_border[type_of_res] + 1,
                                                                ind[2][contractor][type_of_res] // 10))
            if ind[2][contractor][type_of_res] <= 0:
                ind[2][contractor][type_of_res] = 1

            # find and correct all invalidated resource assignments
            for work in range(len(ind[0])):
                if ind[1][work][num_resources] == contractor:
                    ind[1][work][type_of_res] = min(ind[1][work][type_of_res],
                                                    ind[2][contractor][type_of_res])

    return ind


def mate_for_zones(ind1: ChromosomeType, ind2: ChromosomeType,
                   rand: random.Random) -> tuple[ChromosomeType, ChromosomeType]:
    """
    CxOnePoint for zones

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for exchange point selection
    :return: first and second individual
    """
    child1 = Individual(copy_chromosome(ind1))
    child2 = Individual(copy_chromosome(ind2))

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
                work_post_zones[type_of_zone] = rand.randint(0, statuses_available)

    return ind
