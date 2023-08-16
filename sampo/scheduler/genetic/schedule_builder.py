import math
import random
import time
from typing import Callable

import numpy as np
import seaborn as sns
from deap import tools
from deap.base import Toolbox
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame

from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, copy_chromosome, \
    FitnessFunction, TimeFitness, is_chromosome_correct
from sampo.scheduler.native_wrapper import NativeWrapper
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.collections_util import reverse_dictionary


def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   worker_pool: WorkerContractorPool,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec]],
                   rand: random.Random,
                   spec: ScheduleSpec = ScheduleSpec(),
                   work_estimator: WorkTimeEstimator = None,
                   landscape: LandscapeConfiguration = LandscapeConfiguration()) \
        -> tuple[Toolbox, ndarray, ndarray, ndarray]:
    start = time.time()

    # preparing access-optimized data structures
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)
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

    # initial chromosomes construction
    init_chromosomes: dict[str, ChromosomeType] = \
        {name: convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule, spec, order)
        if schedule is not None else None
         for name, (schedule, order, spec) in init_schedules.items()}

    for name, chromosome in init_chromosomes.items():
        if chromosome is not None:
            if not is_chromosome_correct(chromosome, node_indices, parents):
                raise NoSufficientContractorError('HEFTs are deploying wrong chromosomes')

    print(f'Genetic optimizing took {(time.time() - start) * 1000} ms')

    return init_toolbox(wg,
                        contractors,
                        worker_pool,
                        landscape,
                        index2node,
                        work_id2index,
                        worker_name2index,
                        index2contractor,
                        index2contractor_obj,
                        init_chromosomes,
                        mutate_order,
                        mutate_resources,
                        selection_size,
                        rand,
                        spec,
                        worker_pool_indices,
                        contractor2index,
                        contractor_borders,
                        node_indices,
                        parents,
                        Time(0),
                        work_estimator), resources_border, contractors_capacity, resources_min_border


def build_schedule(wg: WorkGraph,
                   contractors: list[Contractor],
                   worker_pool: WorkerContractorPool,
                   population_size: int,
                   generation_number: int,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec]],
                   rand: random.Random,
                   spec: ScheduleSpec,
                   landscape: LandscapeConfiguration = LandscapeConfiguration(),
                   fitness_constructor: Callable[
                       [Callable[[list[ChromosomeType]], list[int]]], FitnessFunction] = TimeFitness,
                   work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                   show_fitness_graph: bool = False,
                   n_cpu: int = 1,
                   assigned_parent_time: Time = Time(0),
                   timeline: Timeline | None = None,
                   time_border: int = None) \
        -> tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]:
    if show_fitness_graph:
        fitness_history = []

    global_start = time.time()

    # preparing access-optimized data structures
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    work_id2index: dict[str, int] = {node.id: index for index, node in enumerate(nodes)}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    contractor2index = {contractor.id: ind for ind, contractor in enumerate(contractors)}
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}

    # construct inseparable_child -> inseparable_parent mapping
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child] = node

    # here we aggregate information about relationships from the whole inseparable chain
    children = {work_id2index[node.id]: list({work_id2index[inseparable_parents[child].id]
                                              for inseparable in node.get_inseparable_chain_with_self()
                                              for child in inseparable.children})
                for node in nodes}

    parents = {work_id2index[node.id]: [] for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].append(node)

    start = time.time()

    toolbox, resources_border, contractors_capacity, resources_min_border = create_toolbox(wg, contractors, worker_pool,
                                                                                           selection_size,
                                                                                           mutate_order,
                                                                                           mutate_resources,
                                                                                           init_schedules,
                                                                                           rand, spec, work_estimator,
                                                                                           landscape)

    native = NativeWrapper(toolbox, wg, contractors, worker_name2index, worker_pool_indices,
                           parents, work_estimator)
    # create population of a given size
    pop = toolbox.population(n=population_size)

    print(f'Toolbox initialization & first population took {(time.time() - start) * 1000} ms')

    if native.native:
        native_start = time.time()
        best_chromosome = native.run_genetic(list([ind[0] for ind in pop]),
                                             mutate_order, mutate_order, mutate_resources, mutate_resources,
                                             mutate_resources, mutate_resources, selection_size)
        print(f'Native evaluated in {(time.time() - native_start) * 1000} ms')
    else:
        # save best individuals
        hof = tools.HallOfFame(1, similar=compare_individuals)

        # probability to participate in mutation and crossover for each individual
        cxpb, mutpb = mutate_order, mutate_order
        mutpb_res, cxpb_res = mutate_resources, mutate_resources

        fitness_f = fitness_constructor(native.evaluate)

        start = time.time()

        # map to each individual fitness function
        pop = [ind for ind in pop if toolbox.validate(ind)]
        fitness = fitness_f.evaluate([ind for ind in pop])

        evaluation_time = time.time() - start

        for ind, fit in zip(pop, fitness):
            ind.fitness.values = [fit]

        hof.update(pop)
        best_fitness = hof[0].fitness.values[0]

        if show_fitness_graph:
            fitness_history.append(sum(fitness) / len(fitness))

        generation = 1
        # the best fitness, track to increase performance by stopping evaluation when not decreasing
        prev_best_fitness = Time.inf()

        print(f'First population evaluation took {(time.time() - start) * 1000} ms')
        start = time.time()

        plateau_steps = 0
        max_plateau_steps = generation_number

        while generation <= generation_number and plateau_steps < max_plateau_steps \
                and (time_border is None or time.time() - global_start < time_border):
            print(f'-- Generation {generation}, population={len(pop)}, best time={best_fitness} --')
            if best_fitness == prev_best_fitness:
                plateau_steps += 1
            else:
                plateau_steps = 0
            prev_best_fitness = best_fitness

            cur_generation = []

            for ind1, ind2 in zip(pop[::2], pop[1::2]):
                # if rand.random() < cxpb:
                child1, child2 = toolbox.mate(ind1, ind2)
                # add to population
                cur_generation.append(child1)
                cur_generation.append(child2)

            if worker_name2index:
                # operations for RESOURCES
                # mutation
                # select types for mutation
                # numbers of changing types
                number_of_type_for_changing = rand.randint(1, len(worker_name2index) - 1)

                # for the crossover, we use those types that did not participate
                # in the mutation(+1 means contractor 'resource')
                # workers_for_mate = list(set(list(range(len(worker_name2index) + 1))) - set(workers))
                # crossover
                # take 2 individuals as input 1 modified individuals

                # workers = rand.sample(range(len(worker_name2index) + 1), number_of_type_for_changing)
                #
                # for ind1, ind2 in zip(pop[:len(pop) // 2], pop[len(pop) // 2:]):
                #     for ind_worker in workers:
                #         # mate resources
                #         if rand.random() < cxpb_res:
                #             child1, child2 = toolbox.mate_resources(ind1, ind2, ind_worker)
                #             # add to population
                #             cur_generation.append(child1)
                #             cur_generation.append(child2)
                #
                #         # mate resource borders
                #         if rand.random() < cxpb_res:
                #             if ind_worker == len(worker_name2index):
                #                 continue
                #             child1, child2 = toolbox.mate_resource_borders(ind1, ind2, ind_worker)
                #             # add to population
                #             cur_generation.append(child1)
                #             cur_generation.append(child2)

                # workers type for changing(+1 means contractor 'resource')
                workers = rand.sample(range(len(worker_name2index) + 1), number_of_type_for_changing)

                # resources mutation
                for worker in workers:
                    low = resources_border[0, worker] if worker != len(worker_name2index) else 0
                    up = resources_border[1, worker] if worker != len(worker_name2index) else 0
                    for mutant in cur_generation:
                        if rand.random() < mutpb_res:
                            toolbox.mutate_resources(mutant, low=low, up=up, type_of_worker=worker)

                # resource borders mutation
                for worker in workers:
                    if worker == len(worker_name2index):
                        continue
                    for mutant in cur_generation:
                        if rand.random() < mutpb_res:
                            toolbox.mutate_resource_borders(mutant,
                                                            contractors_capacity=contractors_capacity,
                                                            resources_min_border=resources_min_border,
                                                            type_of_worker=worker)

            # mutation
            # take 1 individuals as input and return 1 individuals as output
            for mutant in cur_generation:
                toolbox.mutate(mutant)

            evaluation_start = time.time()

            # Gather all the fitness in one list and print the stats
            offspring = [ind for ind in cur_generation if toolbox.validate(ind)]
            print(f'Len offspring: {len(offspring)}')

            offspring_fitness = fitness_f.evaluate(offspring)
            for fit, ind in zip(offspring_fitness, offspring):
                ind.fitness.values = [fit]
            evaluation_time += time.time() - evaluation_start

            if show_fitness_graph:
                _ftn = [f for f in fitness if not math.isinf(f)]
                if len(_ftn) > 0:
                    fitness_history.append(sum(_ftn) / len(_ftn))

            # renewing population
            pop += offspring
            pop = toolbox.select(pop, population_size)
            hof.update(pop)

            best_fitness = hof[0].fitness.values[0]

            generation += 1

        native.close()

        best_chromosome = hof[0]

        # assert that we have valid chromosome
        assert hof[0].fitness.values[0] != Time.inf()

        print(f'Final time: {best_fitness}')
        print(f'Generations processing took {(time.time() - start) * 1000} ms')
        print(f'Evaluation time: {evaluation_time * 1000}')

    scheduled_works, schedule_start_time, timeline, order_nodes = toolbox.chromosome_to_schedule(best_chromosome,
                                                                                                 landscape=landscape,
                                                                                                 timeline=timeline)

    if show_fitness_graph:
        sns.lineplot(
            data=DataFrame.from_records([(generation * 4, v) for generation, v in enumerate(fitness_history)],
                                        columns=['Поколение', 'Функция качества']),
            x='Поколение',
            y='Функция качества',
            palette='r')
        plt.show()

    return {node.id: work for node, work in scheduled_works.items()}, schedule_start_time, timeline, order_nodes


def compare_individuals(first: tuple[ChromosomeType], second: tuple[ChromosomeType]):
    return (first[0] == second[0]).all() and (first[1] == second[1]).all()
