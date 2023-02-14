import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Callable

import dill
import numpy as np
import seaborn as sns
from deap import tools
from deap.base import Toolbox
from deap.tools import initRepeat
from matplotlib import pyplot as plt
from pandas import DataFrame

from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, convert_chromosome_to_schedule
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, Individual, copy_chromosome, \
    FitnessFunction, TimeFitness, init_worker, evaluate
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary


def build_schedule(wg: WorkGraph,
                   contractors: List[Contractor],
                   worker_pool: WorkerContractorPool,
                   population_size: int,
                   generation_number: int,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: Dict[str, tuple[Schedule, list[GraphNode] | None]],
                   rand: random.Random,
                   spec: ScheduleSpec,
                   fitness: Callable[[Toolbox], FitnessFunction] = TimeFitness,
                   work_estimator: WorkTimeEstimator = None,
                   show_fitness_graph: bool = False,
                   n_cpu: int = 1,
                   assigned_parent_time: Time = Time(0),
                   timeline: Timeline | None = None) \
        -> tuple[ScheduleWorkDict, Time, Timeline]:
    """
    Genetic algorithm
    Structure of chromosome:
    [[order of job], [numbers of workers types 1 for each job], [numbers of workers types 2], ... ]
    Different mate and mutation for order and for workers
    Generate order of job by prioritization from HEFTs and from Topological
    Generate resources from min to max
    Overall initial population is valid

    :param show_fitness_graph:
    :param worker_pool:
    :param contractors:
    :param wg:
    :param population_size:
    :param generation_number:
    :param selection_size:
    :param mutate_order:
    :param mutate_resources:
    :param rand:
    :param spec: spec for current scheduling
    :param fitness: the fitness function to be used
    :param init_schedules:
    :param timeline:
    :param n_cpu: number or parallel workers to use in computational process
    :param assigned_parent_time: start time of the whole schedule(time shift)
    :param work_estimator:
    :return: scheduler
    """

    if show_fitness_graph:
        fitness_history = list()

    start = time.time()
    # preparing access-optimized data structures
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

    print(f'Genetic optimizing took {(time.time() - start) * 1000} ms')

    start = time.time()

    # initial chromosomes construction
    init_chromosomes: Dict[str, ChromosomeType] = \
        {name: convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, order)
         for name, (schedule, order) in init_schedules.items()}

    toolbox = init_toolbox(wg, contractors, worker_pool, index2node,
                           work_id2index, worker_name2index, index2contractor,
                           index2contractor_obj, init_chromosomes, mutate_order,
                           mutate_resources, selection_size, rand, spec, worker_pool_indices,
                           contractor2index, contractor_borders, node_indices, index2node_list, parents,
                           assigned_parent_time, work_estimator)

    def prepare_distributed_genetic_args():
        # for more information please refer operators.py#prepare_toolbox
        hyperparams = mutate_order, mutate_resources, selection_size, spec, rand, assigned_parent_time
        return wg, contractors, init_chromosomes, hyperparams

    with ProcessPoolExecutor(max_workers=n_cpu, initializer=init_worker,
                             initargs=(fitness, dill.dumps(work_estimator),
                                       prepare_distributed_genetic_args(),)) as pool:
        # save best individuals
        hof = tools.HallOfFame(1, similar=compare_individuals)
        # create population of a given size
        pop = toolbox.population(n=population_size)

        # probability to participate in mutation and crossover for each individual
        cxpb, mutpb = mutate_order, mutate_order
        mutpb_res, cxpb_res = mutate_resources, mutate_resources

        print(f'Toolbox initialization & first population took {(time.time() - start) * 1000} ms')
        start = time.time()

        # map to each individual fitness function
        fitness = pool.map(evaluate, pop)
        # pool.close()
        # pool.join()
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = [fit]
            ind.fitness.invalid_steps = 1 if fit == Time.inf() else 0

        hof.update(pop)
        best_fitness = hof[0].fitness.values[0]

        if show_fitness_graph:
            fitness_history.append(sum(fitness) / len(fitness))

        g = 0
        # the best fitness, track to increase performance by stopping evaluation when not decreasing
        prev_best_fitness = Time.inf()

        print(f'First population evaluation took {(time.time() - start) * 1000} ms')
        start = time.time()

        invalidation_border = 3
        plateau_steps = 0
        max_plateau_steps = 3

        while g < generation_number and plateau_steps < max_plateau_steps:
            print(f"-- Generation {g}, population={len(pop)}, best time={best_fitness} --")
            if best_fitness == prev_best_fitness:
                plateau_steps += 1
            else:
                plateau_steps = 0
            prev_best_fitness = best_fitness

            # select individuals of next generation
            offspring = toolbox.select(pop, int(math.sqrt(len(pop))))
            # clone selected individuals
            # offspring = [toolbox.clone(ind) for ind in offspring]

            # operations for ORDER
            # crossover
            # take 2 individuals as input 1 modified individuals
            # take after 1: (1,3,5) and (2,4,6) and get pairs 1,2; 3,4; 5,6

            cur_generation = []

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if rand.random() < cxpb:
                    ind1, ind2 = toolbox.mate(child1[0], child2[0])
                    # add to population
                    cur_generation.append(wrap(ind1))
                    cur_generation.append(wrap(ind2))

            # mutation
            # take 1 individuals as input and return 1 individuals as output
            for mutant in offspring:
                if rand.random() < mutpb:
                    ind_order = toolbox.mutate(mutant[0][0])
                    ind = copy_chromosome(mutant[0])
                    ind = (ind_order[0], ind[1], ind[2])
                    # add to population
                    cur_generation.append(wrap(ind))

            # operations for RESOURCES
            # mutation
            # select types for mutation
            # numbers of changing types
            number_of_type_for_changing = rand.randint(1, len(worker_name2index) - 1)
            # workers type for changing(+1 means contractor 'resource')
            workers = rand.sample(range(len(worker_name2index) + 1), number_of_type_for_changing)

            # resources mutation
            for worker in workers:
                low = resources_border[0, worker] if worker != len(worker_name2index) else 0
                up = resources_border[1, worker] if worker != len(worker_name2index) else 0
                for mutant in offspring:
                    if rand.random() < mutpb_res:
                        ind = toolbox.mutate_resources(mutant[0], low=low, up=up, type_of_worker=worker)
                        # add to population
                        cur_generation.append(wrap(ind))

            # resource borders mutation
            for worker in workers:
                if worker == len(worker_name2index):
                    continue
                for mutant in offspring:
                    if rand.random() < mutpb_res:
                        ind = toolbox.mutate_resource_borders(mutant[0],
                                                              contractors_capacity=contractors_capacity,
                                                              resources_min_border=resources_min_border,
                                                              type_of_worker=worker)
                        # add to population
                        cur_generation.append(wrap(ind))

            # for the crossover, we use those types that did not participate in the mutation(+1 means contractor 'resource')
            # workers_for_mate = list(set(list(range(len(worker_name2index) + 1))) - set(workers))
            # crossover
            # take 2 individuals as input 1 modified individuals

            workers = rand.sample(range(len(worker_name2index) + 1), number_of_type_for_changing)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                for ind_worker in workers:
                    # mate resources
                    if rand.random() < cxpb_res:
                        ind1, ind2 = toolbox.mate_resources(child1[0], child2[0], ind_worker)
                        # add to population
                        cur_generation.append(wrap(ind1))
                        cur_generation.append(wrap(ind2))

                    # mate resource borders
                    if rand.random() < cxpb_res:
                        if ind_worker == len(worker_name2index):
                            continue
                        ind1, ind2 = toolbox.mate_resource_borders(child1[0], child2[0], ind_worker)

                        # add to population
                        cur_generation.append(wrap(ind1))
                        cur_generation.append(wrap(ind2))

            # add mutant part of generation to offspring
            offspring.extend(cur_generation)
            cur_generation.clear()
            # Gather all the fitness in one list and print the stats
            invalid_ind = [ind for ind in offspring
                           if ind.fitness.invalid_steps < invalidation_border]
            # for each individual - evaluation
            # print(pool.map(lambda x: x + 2, range(10)))

            invalid_fit = pool.map(evaluate, invalid_ind)
            for fit, ind in zip(invalid_fit, invalid_ind):
                ind.fitness.values = [fit]
                if fit == Time.inf() and ind.fitness.invalid_steps == 0:
                    ind.fitness.invalid_steps = 1

            if show_fitness_graph:
                _ftn = [f for f in fitness if not math.isinf(f)]
                if len(_ftn) > 0:
                    fitness_history.append(sum(_ftn) / len(_ftn))

            def valid(ind: Individual) -> bool:
                if ind.fitness.invalid_steps == 0:
                    return True
                ind.fitness.invalid_steps += 1
                return ind.fitness.invalid_steps < invalidation_border

            # renewing population
            addition = [ind for ind in offspring if valid(ind)]
            print(f'----| Offspring size={len(offspring)}, adding {len(addition)} individuals')
            # pop_size = len(pop)
            # pop = [ind for ind in pop if valid(ind)]
            # print(f'----| Filtered out {pop_size - len(pop)} invalid individuals')
            pop[:] = addition
            hof.update(pop)

            best_fitness = hof[0].fitness.values[0]

            # best = hof[0]
            # fits = [ind.fitness.values[0] for ind in pop]
            # evaluation = chromosome_evaluation(best, index2node, resources_border, work_id2index, worker_name2index,
            #                                   parent2inseparable_son, agents)
            # print("fits: ", fits)
            # print(evaluation)
            g += 1

        chromosome = hof[0][0]

        # assert that we have valid chromosome
        assert hof[0].fitness.values[0] != Time.inf()

        scheduled_works, schedule_start_time, timeline = convert_chromosome_to_schedule(chromosome, worker_pool, index2node,
                                                                                        index2contractor_obj,
                                                                                        worker_pool_indices,
                                                                                        spec, timeline,
                                                                                        assigned_parent_time,
                                                                                        work_estimator)

        print(f'Generations processing took {(time.time() - start) * 1000} ms')

        if show_fitness_graph:
            sns.lineplot(
                data=DataFrame.from_records([(g * 4, v) for g, v in enumerate(fitness_history)],
                                            columns=["Поколение", "Функция качества"]),
                x="Поколение",
                y="Функция качества",
                palette='r')
            plt.show()

        return {node.id: work for node, work in scheduled_works.items()}, schedule_start_time, timeline


def compare_individuals(a: Tuple[ChromosomeType], b: Tuple[ChromosomeType]):
    return (a[0][0] == b[0][0]).all() and (a[0][1] == b[0][1]).all()


def wrap(chromosome: ChromosomeType) -> Individual:
    """
    Created an individual from chromosome

    :param chromosome:
    :return:
    """

    def ind_getter():
        return chromosome

    ind = initRepeat(Individual, ind_getter, n=1)
    ind.fitness.invalid_steps = 0
    return ind
