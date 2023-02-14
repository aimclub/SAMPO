import logging
import multiprocessing as mp
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Dict, Iterable

import dill
import numpy as np
from deap import creator, base, tools
from deap.base import Toolbox

from sampo.scheduler.genetic.converter import convert_chromosome_to_schedule
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ChromosomeType
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool, get_worker_contractor_pool
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary

logger = mp.log_to_stderr(logging.DEBUG)


class FitnessFunction(ABC):

    def __init__(self, tb: Toolbox):
        self._tb = tb

    @abstractmethod
    def evaluate(self, chromosome: ChromosomeType) -> int:
        ...


class TimeFitness(FitnessFunction):

    def __init__(self, tb: Toolbox):
        super().__init__(tb)

    def evaluate(self, chromosome: ChromosomeType) -> int:
        scheduled_works, _, _ = self._tb.chromosome_to_schedule(chromosome)
        finish_time = max(scheduled_works.values(), key=ScheduledWork.finish_time_getter()).finish_time
        return finish_time


class TimeAndResourcesFitness(FitnessFunction):

    def __init__(self, tb: Toolbox):
        super().__init__(tb)

    def evaluate(self, chromosome: ChromosomeType) -> int:
        scheduled_works, _, _ = self._tb.chromosome_to_schedule(chromosome)
        workers_weight = int(np.sum(chromosome[2]))
        return max(scheduled_works.values(), key=ScheduledWork.finish_time_getter()).finish_time + Time(workers_weight)


class DeadlineResourcesFitness(FitnessFunction):

    def __init__(self, deadline: Time, tb: Toolbox):
        super().__init__(tb)
        self._deadline = deadline

    @staticmethod
    def prepare(deadline: Time):
        """
        Returns the constructor of that fitness function prepared to use in Genetic
        """
        return partial(DeadlineResourcesFitness, deadline)

    def evaluate(self, chromosome: ChromosomeType) -> int:
        scheduled_works, _, _ = self._tb.chromosome_to_schedule(chromosome)
        finish_time = max(scheduled_works.values(), key=ScheduledWork.finish_time_getter()).finish_time
        if finish_time > self._deadline:
            return Time.inf()
        else:
            return int(np.sum(chromosome[2]))  # count of workers used in border


# create class FitnessMin, the weights = -1 means that fitness - is function for minimum

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
Individual = creator.Individual


# handle raised errors
def handle_error(error):
    logger.error(error)


def prepare_toolbox(work_estimator: WorkTimeEstimator,  # serialized with dill
                    wg: WorkGraph, contractors: List[Contractor],
                    init_chromosomes: Dict[str, ChromosomeType], genetic_args: tuple):
    # # linking objects, especially with high-cohesion objects like GraphNode
    # # replace order_ids parameter in init_schedules by order_nodes
    # for _, order_ids in init_schedules.values():
    #     for i in range(len(order_ids)):
    #         order_ids[i] = wg[order_ids[i]]

    mutate_order, mutate_resources, selection_size, spec, rand, assigned_parent_time = genetic_args

    # preparing access-optimized data structures
    worker_pool = get_worker_contractor_pool(contractors)
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: Dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: Dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)
    index2node_list = [(index, node) for index, node in enumerate(nodes)]
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}
    node_indices = list(range(len(nodes)))

    contractors_capacity = np.zeros((len(contractors), len(worker_pool)))
    for w_ind, cont2worker in worker_pool_indices.items():
        for c_ind, worker in cont2worker.items():
            contractors_capacity[c_ind][w_ind] = worker.count

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    resources_min_border = np.zeros((len(worker_pool)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count
            resources_min_border[worker_index] = max(resources_min_border[worker_index], req.min_count)

    contractor_borders = np.zeros((len(contractor2index), len(worker_name2index)), dtype=int)
    for ind, contractor in enumerate(contractors):
        for ind_worker, worker in enumerate(contractor.workers.values()):
            contractor_borders[ind, ind_worker] = worker.count

    # construct inseparable_child -> inseparable_parent mapping
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child] = node

    # here we aggregate information about relationships from the whole inseparable chain
    children = {work_id2index[node.id]: [work_id2index[inseparable_parents[child].id]
                                         for inseparable in node.get_inseparable_chain_with_self()
                                         for child in inseparable.children]
                for node in nodes}

    parents = {work_id2index[node.id]: [] for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].append(node)

    return init_toolbox(wg, contractors, worker_pool, index2node,
                        work_id2index, worker_name2index, index2contractor,
                        index2contractor_obj, init_chromosomes, mutate_order,
                        mutate_resources, selection_size, rand, spec, worker_pool_indices,
                        contractor2index, contractor_borders, node_indices, index2node_list, parents,
                        assigned_parent_time, work_estimator)


# initialize worker processes
def init_worker(fitness_constructor, s_work_estimator, genetic_args: tuple):  # serialized):
    # tb, fitness_constructor = dill.loads(serialized)
    # declare scope of a new global variable
    global toolbox
    global fitness_f

    # deserialize with dill
    work_estimator = dill.loads(s_work_estimator)

    # store argument in the global variable for this process
    toolbox = prepare_toolbox(work_estimator, *genetic_args)
    # construct fitness
    fitness_f = fitness_constructor(toolbox)

    logger.info('I\'m here!')


def evaluate(ind) -> Time:
    return toolbox.evaluate(ind)


def evaluation(chromosome):
    return (fitness_f.evaluate(chromosome[0]) if toolbox.validate(chromosome[0]) else Time.inf()).value


def init_toolbox(wg: WorkGraph, contractors: List[Contractor], worker_pool: WorkerContractorPool,
                 index2node: Dict[int, GraphNode],
                 work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                 index2contractor: Dict[int, str],
                 index2contractor_obj: Dict[int, Contractor],
                 init_chromosomes: Dict[str, ChromosomeType],
                 mutate_order: float, mutate_resources: float, selection_size: int,
                 rand: random.Random,
                 spec: ScheduleSpec,
                 worker_pool_indices: dict[int, dict[int, Worker]],
                 contractor2index: dict[str, int],
                 contractor_borders: np.ndarray,
                 node_indices: list[int],
                 index2node_list: list[tuple[int, GraphNode]],
                 parents: Dict[int, list[int]],
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = None) -> base.Toolbox:
    toolbox = base.Toolbox()
    # generate initial population
    toolbox.register("generate_chromosome", generate_chromosome, wg=wg, contractors=contractors,
                     index2node_list=index2node_list, work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders,
                     init_chromosomes=init_chromosomes, rand=rand, work_estimator=work_estimator)

    # create from generate_chromosome function one individual
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.generate_chromosome, n=1)
    # create population from individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # evaluation function
    toolbox.register("evaluate", evaluation)
    # crossover for order
    toolbox.register("mate", mate_scheduling_order, rand=rand)
    # mutation for order. Coefficient luke one or two mutation in individual
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutate_order)
    # selection. Some random individuals and arranges a battle between them as a result in a continuing genus,
    # this is the best among these it
    toolbox.register("select", tools.selTournament, tournsize=selection_size)

    # mutation for resources
    toolbox.register("mutate_resources", mut_uniform_int, probability_mutate_resources=mutate_resources,
                     contractor_count=len(index2contractor), rand=rand)
    # mutation for resource borders
    toolbox.register("mutate_resource_borders", mutate_resource_borders,
                     probability_mutate_contractors=mutate_resources, rand=rand)
    # crossover for resources
    toolbox.register("mate_resources", mate_for_resources, rand=rand)
    # crossover for resource borders
    toolbox.register("mate_resource_borders", mate_for_resource_borders, rand=rand)

    toolbox.register("validate", is_chromosome_correct, node_indices=node_indices, parents=parents)
    toolbox.register("schedule_to_chromosome", convert_schedule_to_chromosome, wg=wg,
                     work_id2index=work_id2index, worker_name2index=worker_name2index,
                     contractor2index=contractor2index, contractor_borders=contractor_borders)
    toolbox.register("chromosome_to_schedule", convert_chromosome_to_schedule, worker_pool=worker_pool,
                     index2node=index2node, index2contractor=index2contractor_obj,
                     worker_pool_indices=worker_pool_indices, spec=spec, assigned_parent_time=assigned_parent_time,
                     work_estimator=work_estimator)
    return toolbox


def copy_chromosome(c: ChromosomeType) -> ChromosomeType:
    return c[0].copy(), c[1].copy(), c[2].copy()


def generate_chromosome(wg: WorkGraph, contractors: List[Contractor], index2node_list: list[tuple[int, GraphNode]],
                        work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                        contractor2index: Dict[str, int], contractor_borders: np.ndarray,
                        init_chromosomes: Dict[str, ChromosomeType], rand: random.Random,
                        work_estimator: WorkTimeEstimator = None) -> ChromosomeType:
    """
    It is necessary to generate valid scheduling, which are satisfied to current dependencies
    That's why will be used the approved order of works (HEFT order and Topological sorting)
    Topological sorts are generating always different
    HEFT is always the same(now not)
    HEFT we will choose in 30% of attempts
    Topological in others

    :param work_estimator:
    :param contractors:
    :param wg:
    :param work_id2index:
    :param index2node_list:
    :param worker_name2index:
    :param contractor2index:
    :param contractor_borders:
    :param rand:
    :param init_chromosomes:
    :return: chromosome
    """
    chance = rand.random()
    if chance < 0.2:
        chromosome = init_chromosomes["heft_end"]
    elif chance < 0.4:
        chromosome = init_chromosomes["heft_between"]
    else:
        schedule = RandomizedTopologicalScheduler(work_estimator,
                                                  int(rand.random() * 1000000)) \
            .schedule(wg, contractors)
        chromosome = convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                                    contractor2index, contractor_borders, schedule)
    return chromosome


def is_chromosome_correct(chromosome: ChromosomeType,
                          node_indices: list[int],
                          parents: Dict[int, list[int]]) -> bool:
    return is_chromosome_order_correct(chromosome, parents) and \
           is_chromosome_contractors_correct(chromosome, node_indices)


def is_chromosome_order_correct(chromosome: ChromosomeType, parents: Dict[int, list[int]]) -> bool:
    work_order = chromosome[0]
    used = set()
    for work_index in work_order:
        used.add(work_index)
        for parent in parents[work_index]:
            if parent not in used:
                return False
    return True


def is_chromosome_contractors_correct(chromosome: ChromosomeType,
                                      work_indices: Iterable[int]) -> bool:
    """
    Checks that assigned contractors can supply assigned workers

    :param chromosome:
    :param contractors_borders:
    :param work_indices:
    :return:
    """
    for work_ind in work_indices:
        resources_count = chromosome[1][:-1, work_ind]
        contractor_ind = chromosome[1][-1, work_ind]
        contractor_border = chromosome[2][contractor_ind]
        for ind, count in enumerate(resources_count):
            if contractor_border[ind] < count:
                return False
    return True


def get_order_tail(head_set: np.ndarray, other: np.ndarray) -> np.ndarray:
    head_set = set(head_set)
    return np.array([node for node in other if node not in head_set])


def mate_scheduling_order(ind1: ChromosomeType, ind2: ChromosomeType, rand: random.Random) \
        -> (ChromosomeType, ChromosomeType):
    """
    Crossover for order
    Basis crossover is cxOnePoint
    But we checked not repeated works in individual order

    :param ind1:
    :param ind2:
    :param rand:
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
    Mutation function for resources
    It changes selected numbers of workers in random work in certain interval for this work

    :param contractor_count:
    :param ind:
    :param low: lower bound specified by `WorkUnit`
    :param up: upper bound specified by `WorkUnit`
    :param type_of_worker:
    :param probability_mutate_resources:
    :param rand:
    :return: mutate individual
    """
    ind = copy_chromosome(ind)

    # select random number from interval from min to max from uniform distribution
    size = len(ind[1][type_of_worker])

    if type_of_worker == len(ind[1]) - 1:
        # print('Contractor mutation!')
        for i in range(size):
            if rand.random() < probability_mutate_resources:
                ind[1][type_of_worker][i] = rand.randint(0, contractor_count - 1)
        return ind

    # change in this interval in random number from interval
    for i, xl, xu in zip(range(size), low, up):
        if rand.random() < probability_mutate_resources:
            # borders
            contractor = ind[1][-1][i]
            border = ind[2][contractor][type_of_worker]
            # TODO Debug why min(xu, border) can be lower than xl
            ind[1][type_of_worker][i] = rand.randint(xl, min(xu, border))

    return ind


def mutate_resource_borders(ind: ChromosomeType, contractors_capacity: np.ndarray, resources_min_border: np.ndarray,
                            type_of_worker: int, probability_mutate_contractors: float, rand: random.Random) \
        -> ChromosomeType:
    """
    Mutation for contractors' resource borders.

    :param ind:
    :param contractors_capacity:
    :param resources_min_border:
    :param type_of_worker:
    :param probability_mutate_contractors:
    :param rand:
    :return:
    """
    ind = copy_chromosome(ind)

    num_contractors = len(ind[2])
    for i in range(num_contractors):
        if rand.random() < probability_mutate_contractors:
            ind[2][i][type_of_worker] -= rand.randint(resources_min_border[type_of_worker] + 1,
                                                      max(resources_min_border[type_of_worker] + 1,
                                                          ind[2][i][type_of_worker] // 10))

    return ind


def mate_for_resources(ind1: ChromosomeType, ind2: ChromosomeType, mate_positions: np.ndarray,
                       rand: random.Random) -> (ChromosomeType, ChromosomeType):
    """
    CxOnePoint for resources

    :param ind1: first individual
    :param ind2: second individual
    :param mate_positions: an array of positions that should be mate
    :param rand: the rand object used for exchange point selection
    :return: first and second individual
    """
    ind1 = copy_chromosome(ind1)
    ind2 = copy_chromosome(ind2)

    # exchange work resources
    res1 = ind1[1][mate_positions]
    res2 = ind2[1][mate_positions]
    cxpoint = rand.randint(1, len(res1))

    mate_positions = rand.sample(list(range(len(res1))), cxpoint)

    res1[mate_positions], res2[mate_positions] = res2[mate_positions], res1[mate_positions]
    return ind1, ind2


def mate_for_resource_borders(ind1: ChromosomeType, ind2: ChromosomeType,
                              mate_positions: np.ndarray, rand: random.Random) -> (ChromosomeType, ChromosomeType):
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
