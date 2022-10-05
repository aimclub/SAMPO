import random
from typing import List, Dict

import numpy as np
from deap import creator, base, tools

from schemas.work_estimator import WorkTimeEstimator
from scheduler.evolution.converter import convert_chromosome_to_schedule
from scheduler.evolution.converter import convert_schedule_to_chromosome, ChromosomeType
from scheduler.topological.base import RandomizedTopologicalScheduler
from schemas.contractor import Contractor, AgentsDict
from schemas.schedule import ScheduledWork
from schemas.time import Time
from schemas.graph import GraphNode, WorkGraph

# create class FitnessMin, the weights = -1 means that fitness - is function for minimum
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
Individual = creator.Individual


def init_toolbox(wg: WorkGraph, contractors: List[Contractor], agents: AgentsDict, index2node: Dict[int, GraphNode],
                 work_id2index: Dict[str, int], worker_name2index: Dict[str, int], resources_border: np.ndarray,
                 init_chromosomes: Dict[str, ChromosomeType],
                 mutate_order: float, mutate_resources: float, selection_size: int,
                 rand: random.Random,
                 work_estimator: WorkTimeEstimator = None) -> base.Toolbox:
    toolbox = base.Toolbox()
    # generate initial population
    toolbox.register("n_per_product", n_per_product, wg=wg, contractors=contractors, index2node=index2node,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator)

    # create from n_per_product function one individual
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.n_per_product, n=1)
    # create population from individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # evaluation function
    toolbox.register("evaluate", chromosome_evaluation, index2node=index2node, resources_border=resources_border,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     agents=agents, work_estimator=work_estimator)
    # crossover for order
    toolbox.register("mate", mate_scheduling_order, rand=rand)
    # mutation for order. Coefficient luke one or two mutation in individual
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.006)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutate_order)
    # selection. Some random individuals and arranges a battle between them as a result in a continuing genus,
    # this is the best among these it
    toolbox.register("select", tools.selTournament, tournsize=len(index2node) // 15)
    toolbox.register("select", tools.selTournament, tournsize=selection_size)

    # mutation for resources
    toolbox.register("mutate_resources", mut_uniform_int, probability_mutate_resources=0.06, rand=rand)
    toolbox.register("mutate_resources", mut_uniform_int, probability_mutate_resources=mutate_resources, rand=rand)
    # crossover for resources
    toolbox.register("mate_resources", mate_for_resources, rand=rand)
    return toolbox


def n_per_product(wg: WorkGraph, contractors: List[Contractor], index2node: Dict[int, GraphNode],
                  work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                  init_chromosomes: Dict[str, ChromosomeType], rand: random.Random,
                  work_estimator: WorkTimeEstimator = None) -> ChromosomeType:
    """
    It is necessary to generate valid scheduling, which are satisfied to current dependencies
    That's why will be used the approved order of works (HEFT order and Topological sorting)
    Topological sorts are generating always different
    HEFT is always the same
    HEFT we will choose in 30% of attempts
    Topological in others
    :param work_estimator:
    :param contractors:
    :param wg:
    :param work_id2index:
    :param index2node:
    :param worker_name2index:
    :param rand:
    :param init_chromosomes:
    :return: chromosome
    """

    # do random choice to choose type of generation current chromosome (HEFT or Topological type)
    chance = rand.random()
    if chance < 0.2:
        chromosome = init_chromosomes["heft_end"]
    elif chance < 0.2:
        chromosome = init_chromosomes["heft_between"]
    else:
        schedule, order = RandomizedTopologicalScheduler(work_estimator,
                                                         int(rand.random() * 1000000)).schedule(wg, contractors)
        chromosome = convert_schedule_to_chromosome(index2node, work_id2index, worker_name2index, schedule, order)
    return chromosome


def chromosome_evaluation(individuals: List[ChromosomeType], index2node: Dict[int, GraphNode],
                          resources_border: np.ndarray,
                          work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                          agents: AgentsDict,
                          work_estimator: WorkTimeEstimator = None) -> Time:
    chromosome = individuals[0]
    if is_chromosome_correct(chromosome, index2node, resources_border):
        scheduled_works = convert_chromosome_to_schedule(chromosome, agents, index2node,
                                                         work_id2index, worker_name2index,
                                                         work_estimator)
        return max(scheduled_works.values(), key=ScheduledWork.finish_time_getter()).finish_time
    else:
        return Time.inf()


def is_chromosome_correct(chromosome: ChromosomeType, index2node: Dict[int, GraphNode],
                          resources_border: np.ndarray) -> bool:
    return is_chromosome_order_correct(chromosome, index2node) & \
           is_chromosome_resources_correct(chromosome, resources_border)


def is_chromosome_order_correct(chromosome: ChromosomeType, index2node: Dict[int, GraphNode]) -> bool:
    work_order = chromosome[0]
    used = set()
    for work_index in work_order:
        node = index2node[work_index]
        for edge in node.edges_from:
            if edge.finish.id in used:
                return False
        used.add(node.id)
    return True


def is_chromosome_resources_correct(chromosome: ChromosomeType, resources_border: np.ndarray) -> bool:
    resources = chromosome[1]
    return (resources_border[0] <= resources).all() and (resources <= resources_border[1]).all()


def mate_scheduling_order(ind1: List[int], ind2: List[int], rand: random.Random) -> (List[int], List[int]):
    """
    Crossover for order
    Basis crossover is cxOnePoint
    But we checked not repeated works in individual order
    :param ind1:
    :param ind2:
    :param rand:
    :return: two cross individuals
    """
    # randomly select the point where the crossover will take place
    crossover_point = rand.randint(1, len(ind1))
    # swapping the second half of both individual
    set_ind = set(ind1)
    ind1[crossover_point:], ind2[crossover_point:] = ind2[crossover_point:], ind1[crossover_point:]

    # check so that everything basic is preserved but there are no repetitive works

    # for first individual
    set_first_part = set(ind1[:crossover_point])
    set_second_part = set(ind1[crossover_point:])
    # no, these works in the result
    set_free = list(set_ind - set_first_part - set_second_part)
    # put them in the second part
    for i in range(len(ind1[crossover_point:])):
        if ind1[crossover_point:][i] in set_first_part:
            ind1[crossover_point + i] = set_free[0]
            set_free.pop(0)

    # for second individual the same
    set_first_part = set(ind2[:crossover_point])
    set_second_part = set(ind2[crossover_point:])
    set_free = list(set_ind - set_first_part - set_second_part)

    for i in range(len(ind2[crossover_point:])):
        if ind2[crossover_point:][i] in set_first_part:
            ind2[crossover_point + i] = set_free[0]
            set_free.pop(0)

    # just in case, check that the work is not repeat
    assert (len(ind1) == len(set(ind1)))
    assert (len(ind2) == len(set(ind2)))

    return ind1, ind2


def mut_uniform_int(chromosome: ChromosomeType, low: np.ndarray, up: np.ndarray, type_of_worker: int,
                    probability_mutate_resources: float, rand: random.Random) -> ChromosomeType:
    """
    Mutation function for resources
    It changes selected numbers of workers in random work in certain interval for this work
    :param chromosome:
    :param low:
    :param up:
    :param type_of_worker:
    :param probability_mutate_resources:
    :param rand:
    :return: mutate individual
    """
    # select random number from interval from min to max from uniform distribution
    size = len(chromosome[1][type_of_worker])

    # change in this interval in random number from interval
    for i, xl, xu in zip(range(size), low, up):
        if rand.random() < probability_mutate_resources:
            chromosome[1][type_of_worker][i] = rand.randint(xl, xu)
    return chromosome


def mate_for_resources(ind1: np.ndarray, ind2: np.ndarray, rand: random.Random) -> (
        np.ndarray, np.ndarray):
    """
    CxOnePoint for resources
    :param ind1:
    :param ind2:
    :param rand:
    :return:
    """
    cxpoint = rand.randint(1, len(ind1))
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2
