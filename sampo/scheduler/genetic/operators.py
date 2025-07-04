import math
import random
from copy import deepcopy
from operator import attrgetter
from typing import Callable, Iterable

import numpy as np
from deap import base, tools
from deap.base import Toolbox

from sampo.api.genetic_api import ChromosomeType, FitnessFunction, Individual
from sampo.scheduler.genetic.converter import (convert_schedule_to_chromosome, convert_chromosome_to_schedule,
                                               ScheduleGenerationScheme)
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler
from sampo.scheduler.lft.base import RandomizedLFTScheduler
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
    """
    Fitness function that relies on finish time.
    """
    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) \
            -> tuple[int | float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value, )
        return (schedule.execution_time.value, )


class SumOfResourcesPeaksFitness(FitnessFunction):
    """
    Fitness function that relies on sum of resources peaks usage.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[float]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value, )
        return (resources_peaks_sum(schedule, self._resources_names), )


class SumOfResourcesFitness(FitnessFunction):
    """
    Fitness function that relies on sum of resources usage.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> float:
        schedule = evaluator(chromosome)
        if schedule is None:
            return Time.inf().value
        return resources_sum(schedule, self._resources_names)


class TimeWithResourcesFitness(FitnessFunction):
    """
    Fitness function that relies on finish time and the set of resources.
    """

    def __init__(self, resources_names: Iterable[str] | None = None):
        self._resources_names = list(resources_names) if resources_names is not None else None

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> float:
        schedule = evaluator(chromosome)
        if schedule is None:
            return Time.inf().value
        return schedule.execution_time.value + resources_peaks_sum(schedule, self._resources_names)


class DeadlineResourcesFitness(FitnessFunction):
    """
    The fitness function is dependent on the set of resources and requires the end time to meet the deadline.
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
            return (Time.inf().value, )
        return (resources_peaks_sum(schedule, self._resources_names) \
                    * max(1.0, schedule.execution_time.value / self._deadline.value), )


class DeadlineCostFitness(FitnessFunction):
    """
    The fitness function is dependent on the cost of resources and requires the end time to meet the deadline.
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
            return (Time.inf().value, )
        return (resources_costs_sum(schedule, self._resources_names) \
                * max(1.0, schedule.execution_time.value / self._deadline.value), )


class TimeAndResourcesFitness(FitnessFunction):
    """
    Bi-objective fitness function of finish time and sum of resources peaks.
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
                 parents: dict[int, set[int]],
                 children: dict[int, set[int]],
                 resources_border: np.ndarray,
                 assigned_parent_time: Time = Time(0),
                 fitness_weights: tuple[int | float, ...] = (-1,),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                 only_lft_initialization: bool = False,
                 is_multiobjective: bool = False) -> base.Toolbox:
    """
    Object, that include set of functions (tools) for genetic model and other functions related to it.
    list of parameters that received this function is sufficient and complete to manipulate with genetic algorithm

    :return: Object, included tools for genetic algorithm
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
    toolbox.register('mate', mate, rand=rand, toolbox=toolbox)
    # combined mutation
    toolbox.register('mutate', mutate, order_mutpb=mut_order_pb, res_mutpb=mut_res_pb, zone_mutpb=mut_zone_pb,
                     rand=rand, parents=parents, children=children, resources_border=resources_border,
                     statuses_available=statuses_available)
    # crossover for order
    toolbox.register('mate_order', mate_scheduling_order, rand=rand, toolbox=toolbox)
    # mutation for order
    toolbox.register('mutate_order', mutate_scheduling_order, mutpb=mut_order_pb, rand=rand, parents=parents,
                     children=children)
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
                     contractor_borders=contractor_borders)
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
    if toolbox.validate(chromosome):
        sworks = toolbox.chromosome_to_schedule(chromosome)[0]
        return Schedule.from_scheduled_works(sworks.values(), wg)
    else:
        return None


def register_individual_constructor(fitness_weights: tuple[int | float, ...], toolbox: base.Toolbox):
    class IndividualFitness(base.Fitness):
        weights = fitness_weights

    toolbox.register('Individual', Individual.prepare(IndividualFitness))


def copy_individual(ind: Individual, toolbox: Toolbox) -> Individual:
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
    """
    Generates n chromosomes.
    Do not use `generate_chromosome` function.
    """

    def randomized_init(is_topological: bool = False) -> ChromosomeType:
        if is_topological:
            schedule = RandomizedTopologicalScheduler(work_estimator, int(rand.random() * 1000000)) \
                .schedule(wg, contractors, spec, landscape=landscape)[0]
        else:
            schedule = RandomizedLFTScheduler(work_estimator=work_estimator, rand=rand).schedule(wg, contractors, spec,
                                                                                                 landscape=landscape)[0]
        return convert_schedule_to_chromosome(work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, spec, landscape)

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
            .schedule(wg, contractors, spec, landscape=landscape)[0]
        return convert_schedule_to_chromosome(work_id2index, worker_name2index,
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

    return chromosome


def select_new_population(population: list[Individual], k: int) -> list[Individual]:
    """
    Selection operator for genetic algorithm.
    Select top k individuals in population.
    """
    population = sorted(population, key=attrgetter('fitness'), reverse=True)
    return population[:k]


def is_chromosome_correct(ind: Individual, node_indices: list[int], parents: dict[int, set[int]],
                          contractor_borders: np.ndarray) -> bool:
    """
    Check correctness of works order and contractors borders.
    """
    return is_chromosome_order_correct(ind, parents) and \
        is_chromosome_contractors_correct(ind, node_indices, contractor_borders)


def is_chromosome_order_correct(ind: Individual, parents: dict[int, set[int]]) -> bool:
    """
    Checks that assigned order of works are topologically correct.
    """
    work_order = ind[0]
    used = set()
    for work_index in work_order:
        used.add(work_index)
        if not parents[work_index].issubset(used):
            # logger.error(f'Order validation failed: {work_order}')
            return False
    return True


def is_chromosome_contractors_correct(ind: Individual, work_indices: Iterable[int],
                                      contractor_borders: np.ndarray) -> bool:
    """
    Checks that assigned contractors can supply assigned workers.
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
    """
    Get a new part in topologic order for chromosome.
    This function is needed to make crossover for order.
    """
    order = set(order)
    return np.array([node for node in other_order if node not in order])


def mate_scheduling_order(ind1: Individual, ind2: Individual, rand: random.Random,
                          toolbox: Toolbox, copy: bool = True) -> tuple[Individual, Individual]:
    """
    Two-Point crossover for order.

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for randomized operations
    :param toolbox: toolbox
    :param copy: if True individuals will be copied before mating so as not to change them

    :return: two mated individuals
    """
    child1, child2 = (toolbox.copy_individual(ind1), toolbox.copy_individual(ind2)) if copy else (ind1, ind2)

    order1, order2 = child1[0], child2[0]
    parent1 = ind1[0].copy()

    min_mating_amount = len(order1) // 4

    two_point_order_crossover(order1, order2, min_mating_amount, rand)
    two_point_order_crossover(order2, parent1, min_mating_amount, rand)

    return toolbox.Individual(child1), toolbox.Individual(child2)


def two_point_order_crossover(child: np.ndarray, other_parent: np.ndarray, min_mating_amount: int, rand: random.Random):
    """
    This faction realizes Two-Point crossover for order.

    :param child: order to which implements crossover, it is equal to order of first parent.
    :param other_parent: order of second parent from which mating part will be taken.
    :param min_mating_amount: minimum amount of mating part
    :param rand: the rand object used for randomized operations

    :return: child mated with other parent
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


def mutate_scheduling_order(ind: Individual, mutpb: float, rand: random.Random,
                            parents: dict[int, set[int]], children: dict[int, set[int]]) -> Individual:
    """
    Mutation operator for works scheduling order.

    :param ind: the individual to be mutated
    :param mutpb: probability of gene mutation
    :param rand: the rand object used for randomized operations
    :param parents: mapping object of works and their parent-works to create valid order
    :param children: mapping object of works and their children-works to create valid order

    :return: mutated individual
    """
    order = ind[0]
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
                              initial=i - 1) + 1
            # find min index of child of the current work
            # +i because the slice [i + 1:] was taken, and +1 is not needed because these indexes will be shifted left
            # after current work deletion
            i_children = np.min(np.where(np.isin(order[i + 1:], list(children[work]), assume_unique=True))[0],
                                initial=0) + i
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

    return ind


def mate_resources(ind1: Individual, ind2: Individual, rand: random.Random,
                   optimize_resources: bool, toolbox: Toolbox, copy: bool = True) -> tuple[Individual, Individual]:
    """
    One-Point crossover for resources.

    :param ind1: first individual
    :param ind2: second individual
    :param optimize_resources: if True resource borders should be changed after mating
    :param rand: the rand object used for randomized operations
    :param copy: if True individuals will be copied before mating so as not to change them
    :param toolbox: toolbox

    :return: two mated individuals
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


def mate(ind1: Individual, ind2: Individual, optimize_resources: bool, rand: random.Random, toolbox: Toolbox) \
        -> tuple[Individual, Individual]:
    """
    Combined crossover function of Two-Point crossover for order, One-Point crossover for resources
    and One-Point crossover for zones.

    :param ind1: first individual
    :param ind2: second individual
    :param optimize_resources: if True resource borders should be changed after mating
    :param rand: the rand object used for randomized operations
    :param toolbox: toolbox

    :return: two mated individuals
    """
    child1, child2 = mate_scheduling_order(ind1, ind2, rand, toolbox, copy=True)
    child1, child2 = mate_resources(child1, child2, rand, optimize_resources, toolbox, copy=False)
    # TODO Make better crossover for zones and uncomment this
    # child1, child2 = mate_for_zones(child1, child2, rand, copy=False)

    return toolbox.Individual(child1), toolbox.Individual(child2)


def mutate(ind: Individual, resources_border: np.ndarray, parents: dict[int, set[int]],
           children: dict[int, set[int]], statuses_available: int,
           order_mutpb: float, res_mutpb: float, zone_mutpb: float,
           rand: random.Random) -> Individual:
    """
    Combined mutation function of mutation for order, mutation for resources and mutation for zones.

    :param ind: the individual to be mutated
    :param resources_border: low and up borders of resources amounts
    :param parents: mapping object of works and their parent-works to create valid order
    :param children: mapping object of works and their children-works to create valid order
    :param statuses_available: number of statuses available
    :param order_mutpb: probability of order's gene mutation
    :param res_mutpb: probability of resources' gene mutation
    :param zone_mutpb: probability of zones' gene mutation
    :param rand: the rand object used for randomized operations

    :return: mutated individual
    """
    mutant = mutate_scheduling_order(ind, order_mutpb, rand, parents, children)
    mutant = mutate_resources(mutant, res_mutpb, rand, resources_border)
    # TODO Make better mutation for zones and uncomment this
    # mutant = mutate_for_zones(mutant, statuses_available, zone_mutpb, rand)

    return mutant


def mutate_resource_borders(ind: Individual, mutpb: float, rand: random.Random,
                            contractor_borders: np.ndarray) -> Individual:
    """
    Mutation function for contractors' resource borders.

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
    """
    Changes numeric values in m x n part of chromosome.
    This function is needed to make mutation for resources and resource borders.
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
    """
    CxOnePoint for zones

    :param ind1: first individual
    :param ind2: second individual
    :param rand: the rand object used for randomized operations
    :param copy: if True individuals will be copied before mating so as not to change them
    :param toolbox: toolbox

    :return: two mated individuals
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
    """
    Mutation function for zones.
    It changes selected numbers of zones in random work in a certain interval from available statuses.

    :param ind: the individual to be mutated
    :param mutpb: probability of gene mutation
    :param rand: the rand object used for randomized operations
    :param statuses_available: number of statuses available

    :return: mutated individual
    """
    # select random number from interval from min to max from uniform distribution
    zones = ind[4]
    if zones.size:
        mask = np.array([[rand.random() < mutpb for _ in range(zones.shape[1])] for _ in range(zones.shape[0])])
        new_zones = np.array([rand.randint(0, statuses_available - 1) for _ in range(mask.sum())])
        zones[mask] = new_zones

    return ind
