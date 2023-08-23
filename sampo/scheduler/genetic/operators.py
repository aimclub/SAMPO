import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Iterable, Callable

import numpy as np
from deap import creator, base, tools
from deap.tools import initRepeat

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
                 index2contractor: dict[int, str],
                 index2contractor_obj: dict[int, Contractor],
                 init_chromosomes: dict[str, tuple[ChromosomeType, float]],
                 mutate_order: float,
                 mutate_resources: float,
                 selection_size: int,
                 rand: random.Random,
                 spec: ScheduleSpec,
                 worker_pool_indices: dict[int, dict[int, Worker]],
                 contractor2index: dict[str, int],
                 contractor_borders: np.ndarray,
                 resources_min_border: np.ndarray,
                 node_indices: list[int],
                 parents: dict[int, list[int]],
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> base.Toolbox:
    """
    Object, that include set of functions (tools) for genetic model and other functions related to it.
    list of parameters that received this function is sufficient and complete to manipulate with genetic

    :return: Object, included tools for genetic
    """
    toolbox = base.Toolbox()
    # generate initial population
    toolbox.register('generate_chromosome', generate_chromosome, wg=wg, contractors=contractors,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape)

    # create from generate_chromosome function one individual
    toolbox.register('individual', tools.initRepeat, Individual, toolbox.generate_chromosome, n=1)
    # create population from individuals
    toolbox.register('population', generate_population, wg=wg, contractors=contractors, spec=spec,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator, landscape=landscape)
    # crossover for order
    toolbox.register('mate', mate_scheduling_order, rand=rand)
    # mutation for order. Coefficient luke one or two mutation in individual
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=mutate_order)
    # selection. Some random individuals and arranges a battle between them as a result in a continuing genus,
    # this is the best among these it
    toolbox.register('select', tools.selTournament, tournsize=selection_size)

    # mutation for resources
    toolbox.register('mutate_resources', mut_uniform_int, probability_mutate_resources=mutate_resources,
                     contractor_count=len(index2contractor), rand=rand)
    # mutation for resource borders
    toolbox.register('mutate_resource_borders', mutate_resource_borders,
                     probability_mutate_contractors=mutate_resources, rand=rand,
                     contractors_capacity=contractor_borders, resources_min_border=resources_min_border)
    # crossover for resources
    toolbox.register('mate_resources', mate_for_resources, rand=rand)
    # crossover for resource borders
    toolbox.register('mate_resource_borders', mate_for_resource_borders, rand=rand)

    toolbox.register('validate', is_chromosome_correct, node_indices=node_indices, parents=parents)
    toolbox.register('schedule_to_chromosome', convert_schedule_to_chromosome, wg=wg,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders, spec=spec)
    toolbox.register("chromosome_to_schedule", convert_chromosome_to_schedule, worker_pool=worker_pool,
                     index2node=index2node, index2contractor=index2contractor_obj,
                     worker_pool_indices=worker_pool_indices, assigned_parent_time=assigned_parent_time,
                     work_estimator=work_estimator, worker_name2index=worker_name2index,
                     contractor2index=contractor2index,
                     landscape=landscape)
    return toolbox


def copy_chromosome(chromosome: ChromosomeType) -> ChromosomeType:
    return chromosome[0].copy(), chromosome[1].copy(), chromosome[2].copy(), deepcopy(chromosome[3])


def wrap(chromosome: ChromosomeType) -> Individual:
    """
    Created an individual from chromosome.
    """

    def ind_getter():
        return chromosome

    ind = initRepeat(Individual, ind_getter, n=1)
    ind.fitness.invalid_steps = 0
    return ind


def generate_population(size_population: int,
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
                        landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
    """
    Generates population using chromosome weights.
    Do not use `generate_chromosome` function.
    """
    def randomized_init() -> ChromosomeType:
        schedule = RandomizedTopologicalScheduler(work_estimator,
                                                  int(rand.random() * 1000000)) \
            .schedule(wg, contractors, spec, landscape=landscape)
        return convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, spec)

    # chromosome types' weights
    # these numbers are the probability weights: prob = norm(weights), sum(prob) = 1
    weights = [2, 2, 1, 1, 1, 1, 2]

    for i, (_, importance, _) in enumerate(init_chromosomes.values()):
        weights[i] = int(weights[i] * importance)

    weights_multiplier = math.ceil(size_population / sum(weights))

    for i in range(len(weights)):
        weights[i] *= weights_multiplier

    all_types = ['heft_end', 'heft_between', '12.5%', '25%', '75%', '87.5%', 'topological']
    chromosome_types = rand.sample(all_types, k=size_population, counts=weights)

    chromosomes = [init_chromosomes[generated_type][0] if generated_type != 'topological' else None
                   for generated_type in chromosome_types]
    for i, chromosome in enumerate(chromosomes):
        if chromosome is None:
            chromosomes[i] = randomized_init()

    return [wrap(chromosome) for chromosome in chromosomes]


def generate_chromosome(wg: WorkGraph,
                        contractors: list[Contractor],
                        work_id2index: dict[str, int],
                        worker_name2index: dict[str, int],
                        contractor2index: dict[str, int],
                        contractor_borders: np.ndarray,
                        init_chromosomes: dict[str, tuple[ChromosomeType, float]],
                        spec: ScheduleSpec,
                        rand: random.Random,
                        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                        landscape: LandscapeConfiguration = LandscapeConfiguration()) -> ChromosomeType:
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
                                              contractor2index, contractor_borders, schedule, spec)

    chromosome = None
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

    if chromosome is None:
        chromosome = randomized_init()

    return chromosome


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
    ind1 = copy_chromosome(ind1)
    ind2 = copy_chromosome(ind2)

    order1 = ind1[0]
    order2 = ind2[0]

    # randomly select the point where the crossover will take place
    crossover_point = rand.randint(1, len(ind1))

    ind1_new_tail = get_order_tail(order1[:crossover_point], order2)
    ind2_new_tail = get_order_tail(order2[:crossover_point], order1)

    order1[crossover_point:] = ind1_new_tail
    order2[crossover_point:] = ind2_new_tail

    return ind1, ind2


def mut_uniform_int(ind: ChromosomeType, low: np.ndarray, up: np.ndarray, type_of_worker: int,
                    probability_mutate_resources: float, contractor_count: int, rand: random.Random) -> ChromosomeType:
    """
    Mutation function for resources.
    It changes selected numbers of workers in random work in a certain interval for this work.

    :param low: lower bound specified by `WorkUnit`
    :param up: upper bound specified by `WorkUnit`
    :return: mutate individual
    """
    ind = copy_chromosome(ind)

    # select random number from interval from min to max from uniform distribution
    size = len(ind[1])
    workers_count = len(ind[1][0])

    if type_of_worker == workers_count - 1:
        # print('Contractor mutation!')
        for i in range(size):
            if rand.random() < probability_mutate_resources:
                ind[1][i][type_of_worker] = rand.randint(0, contractor_count - 1)
        return ind

    # change in this interval in random number from interval
    for i, xl, xu in zip(range(size), low, up):
        if rand.random() < probability_mutate_resources:
            # borders
            contractor = ind[1][i][-1]
            border = ind[2][contractor][type_of_worker]
            # TODO Debug why min(xu, border) can be lower than xl
            ind[1][i][type_of_worker] = rand.randint(xl, max(xl, min(xu, border)))

    return ind


def mutate_resource_borders(ind: ChromosomeType, contractors_capacity: np.ndarray, resources_min_border: np.ndarray,
                            type_of_worker: int, probability_mutate_contractors: float, rand: random.Random) \
        -> ChromosomeType:
    """
    Mutation for contractors' resource borders.
    """
    ind = copy_chromosome(ind)

    num_resources = len(resources_min_border)
    num_contractors = len(ind[2])
    for contractor in range(num_contractors):
        if rand.random() < probability_mutate_contractors:
            ind[2][contractor][type_of_worker] -= rand.randint(resources_min_border[type_of_worker] + 1,
                                                               max(resources_min_border[type_of_worker] + 1,
                                                                   ind[2][contractor][type_of_worker] // 10))
            if ind[2][contractor][type_of_worker] <= 0:
                ind[2][contractor][type_of_worker] = 1

            # find and correct all invalidated resource assignments
            for work in range(len(ind[0])):
                if ind[1][work][num_resources] == contractor:
                    ind[1][work][type_of_worker] = min(ind[1][work][type_of_worker],
                                                       ind[2][contractor][type_of_worker])

    return ind


def mate_for_resources(ind1: ChromosomeType, ind2: ChromosomeType, mate_positions: np.ndarray,
                       rand: random.Random) -> (ChromosomeType, ChromosomeType):
    """
    CxOnePoint for resources.

    :param ind1: first individual
    :param ind2: second individual
    :param mate_positions: an array of positions that should be mate
    :param rand: the rand object used for exchange point selection
    :return: first and second individual
    """
    ind1 = copy_chromosome(ind1)
    ind2 = copy_chromosome(ind2)

    # exchange work resources
    res1 = ind1[1][:, mate_positions]
    res2 = ind2[1][:, mate_positions]
    cxpoint = rand.randint(1, len(res1))

    mate_positions = rand.sample(list(range(len(res1))), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]
    return ind1, ind2


def mate_for_resource_borders(ind1: ChromosomeType, ind2: ChromosomeType,
                              mate_positions: np.ndarray, rand: random.Random) -> (ChromosomeType, ChromosomeType):
    """
    Crossover for contractors' resource borders.
    """
    ind1 = copy_chromosome(ind1)
    ind2 = copy_chromosome(ind2)

    num_contractors = len(ind1[2])
    contractors_to_mate = rand.sample(list(range(num_contractors)), rand.randint(1, num_contractors))

    if rand.randint(0, 2) == 0:
        # trying to mate whole contractors
        border1 = ind1[2][contractors_to_mate]
        border2 = ind2[2][contractors_to_mate]
        border1[:], border2[:] = border2[:], border1[:]
    else:
        # trying to mate part of selected contractors
        border1 = ind1[2][contractors_to_mate]
        border2 = ind2[2][contractors_to_mate]
        for c_border1, c_border2 in zip(border1, border2):
            # mate_positions = rand.sample(list(range(len(c_border1))), rand.randint(1, len(c_border1)))
            c_border1[mate_positions], c_border2[mate_positions] = c_border2[mate_positions], c_border1[mate_positions]

    return ind1, ind2
