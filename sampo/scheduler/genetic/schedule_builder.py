import random
import time
from typing import Callable

import numpy as np
from deap import tools
from deap.base import Toolbox

from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, FitnessFunction, TimeFitness
from sampo.scheduler.native_wrapper import NativeWrapper
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.schemas.resources import Worker


def create_toolbox_and_mapping_objects(wg: WorkGraph,
                                       contractors: list[Contractor],
                                       worker_pool: WorkerContractorPool,
                                       population_size: int,
                                       mutate_order: float,
                                       mutate_resources: float,
                                       mutate_zones: float,
                                       init_schedules: dict[
                                           str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                                       rand: random.Random,
                                       spec: ScheduleSpec = ScheduleSpec(),
                                       fitness_weights: tuple[int | float] = (-1,),
                                       work_estimator: WorkTimeEstimator = None,
                                       sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                                       assigned_parent_time: Time = Time(0),
                                       landscape: LandscapeConfiguration = LandscapeConfiguration(),
                                       only_lft_initialization: bool = False,
                                       is_multiobjective: bool = False,
                                       verbose: bool = True) \
        -> tuple[Toolbox, dict[str, int], dict[int, dict[int, Worker]], dict[int, set[int]]]:
    start = time.time()

    # preparing access-optimized data structures
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    index2zone = {ind: zone for ind, zone in enumerate(landscape.zone_config.start_statuses)}
    contractor2index = {contractor.id: ind for ind, contractor in enumerate(contractors)}
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}
    node_indices = list(range(len(nodes)))

    # construct inseparable_child -> inseparable_parent mapping
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child] = node

    # here we aggregate information about relationships from the whole inseparable chain
    children = {work_id2index[node.id]: set([work_id2index[inseparable_parents[child].id]
                                             for inseparable in node.get_inseparable_chain_with_self()
                                             for child in inseparable.children])
                for node in nodes}

    parents = {work_id2index[node.id]: set() for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].add(node)

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count

    contractor_borders = np.zeros((len(contractor2index), len(worker_name2index)), dtype=int)
    for ind, contractor in enumerate(contractors):
        for ind_worker, worker in enumerate(contractor.workers.values()):
            contractor_borders[ind, ind_worker] = worker.count

    init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]] = \
        {name: (convert_schedule_to_chromosome(work_id2index, worker_name2index,
                                               contractor2index, contractor_borders, schedule, chromosome_spec,
                                               landscape, order),
                importance, chromosome_spec)
        if schedule is not None else None
         for name, (schedule, order, chromosome_spec, importance) in init_schedules.items()}

    if verbose:
        print(f'Genetic optimizing took {(time.time() - start) * 1000} ms')

    return init_toolbox(wg,
                        contractors,
                        worker_pool,
                        landscape,
                        index2node,
                        work_id2index,
                        worker_name2index,
                        index2contractor_obj,
                        index2zone,
                        init_chromosomes,
                        mutate_order,
                        mutate_resources,
                        mutate_zones,
                        landscape.zone_config.statuses.statuses_available(),
                        population_size,
                        rand,
                        spec,
                        worker_pool_indices,
                        contractor2index,
                        contractor_borders,
                        node_indices,
                        parents,
                        children,
                        resources_border,
                        assigned_parent_time,
                        fitness_weights,
                        work_estimator,
                        sgs_type,
                        only_lft_initialization,
                        is_multiobjective), worker_name2index, worker_pool_indices, parents


def build_schedules(wg: WorkGraph,
                    contractors: list[Contractor],
                    worker_pool: WorkerContractorPool,
                    population_size: int,
                    generation_number: int,
                    mutpb_order: float,
                    mutpb_res: float,
                    mutpb_zones: float,
                    init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                    rand: random.Random,
                    spec: ScheduleSpec,
                    landscape: LandscapeConfiguration = LandscapeConfiguration(),
                    fitness_constructor: Callable[
                        [Callable[[list[ChromosomeType]], list[Schedule]]], FitnessFunction] = TimeFitness,
                    fitness_weights: tuple[int | float] = (-1,),
                    work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                    sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                    n_cpu: int = 1,
                    assigned_parent_time: Time = Time(0),
                    timeline: Timeline | None = None,
                    time_border: int | None = None,
                    max_plateau_steps: int | None = None,
                    optimize_resources: bool = False,
                    deadline: Time | None = None,
                    only_lft_initialization: bool = False,
                    is_multiobjective: bool = False,
                    verbose: bool = True) \
        -> list[tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]]:
    """
    Genetic algorithm.
    Structure of chromosome:
    [[order of job],
     [[numbers of workers types 1 for each job], [numbers of workers types 2], ... ],
      [[border of workers types 1 for each contractor], [border of workers types 2], ...]
    ]

    Different mate and mutation for order and for workers.
    Generate order of job by prioritization from HEFTs and from Topological.
    Generate resources from min to max.
    Overall initial population is valid.

    :return: schedule
    """
    global_start = start = time.time()

    toolbox, *mapping_objects = create_toolbox_and_mapping_objects(wg, contractors, worker_pool, population_size,
                                                                   mutpb_order, mutpb_res, mutpb_zones, init_schedules,
                                                                   rand, spec, fitness_weights, work_estimator,
                                                                   sgs_type, assigned_parent_time, landscape,
                                                                   only_lft_initialization, is_multiobjective,
                                                                   verbose)

    worker_name2index, worker_pool_indices, parents = mapping_objects

    native = NativeWrapper(toolbox, wg, contractors, worker_name2index, worker_pool_indices,
                           parents, work_estimator)
    # create population of a given size
    chromosomes = toolbox.population_chromosomes(n=population_size)

    if verbose:
        print(f'Toolbox initialization & first population took {(time.time() - start) * 1000} ms')

    if native.native:
        native_start = time.time()
        best_chromosomes = [native.run_genetic(chromosomes, mutpb_order, mutpb_order, mutpb_res, mutpb_res, mutpb_res,
                                               mutpb_res, population_size)]
        if verbose:
            print(f'Native evaluated in {(time.time() - native_start) * 1000} ms')
    else:
        have_deadline = deadline is not None
        # save best individuals
        hof = tools.ParetoFront(similar=compare_individuals)

        fitness_f = fitness_constructor(native.evaluate) if not have_deadline else TimeFitness(native.evaluate)
        if have_deadline:
            toolbox.register_individual_constructor((-1,))
        pop = [toolbox.Individual(chromosome) for chromosome in chromosomes if toolbox.validate(chromosome)]

        evaluation_start = time.time()

        # map to each individual fitness function
        fitness = fitness_f.evaluate(pop)

        evaluation_time = time.time() - evaluation_start

        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        hof.update(pop)
        best_fitness = hof[0].fitness.values

        if verbose:
            print(f'First population evaluation took {evaluation_time * 1000} ms')

        start = time.time()

        generation = 1
        plateau_steps = 0
        new_generation_number = generation_number if not have_deadline else generation_number // 2
        max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number

        while generation <= new_generation_number and plateau_steps < max_plateau_steps \
                and (time_border is None or time.time() - global_start < time_border):
            if verbose:
                print(f'-- Generation {generation}, population={len(pop)}, best fitness={best_fitness} --')

            rand.shuffle(pop)

            offspring_chromosomes = make_offspring(toolbox, pop, optimize_resources)
            offspring = [toolbox.Individual(chromosome) for chromosome in offspring_chromosomes]

            evaluation_start = time.time()

            offspring_fitness = fitness_f.evaluate(offspring)

            for ind, fit in zip(offspring, offspring_fitness):
                ind.fitness.values = fit

            evaluation_time += time.time() - evaluation_start

            # renewing population
            pop += offspring
            pop = toolbox.select(pop)
            hof.update(pop)

            prev_best_fitness = best_fitness
            best_fitness = hof[0].fitness.values
            plateau_steps = plateau_steps + 1 if best_fitness == prev_best_fitness else 0

            if have_deadline and best_fitness[0] <= deadline:
                if all([ind.fitness.values[0] <= deadline for ind in pop]):
                    break

            generation += 1

        # Second stage to optimize resources if deadline is assigned

        if have_deadline:

            fitness_resource_f = fitness_constructor(native.evaluate)
            toolbox.register_individual_constructor(fitness_weights)
            # clear best individuals
            hof.clear()

            for ind in pop:
                ind.time = ind.fitness.values[0]

            if best_fitness[0] > deadline:
                print(f'Deadline not reached !!! Deadline {deadline} < best time {best_fitness[0]}')
                pop = [ind for ind in pop if ind.time == best_fitness[0]]
            else:
                optimize_resources = True
                pop = [ind for ind in pop if ind.time <= deadline]

            new_pop = []
            for ind in pop:
                ind_time = ind.time
                new_ind = toolbox.copy_individual(ind)
                new_ind.time = ind_time
                new_pop.append(new_ind)
            del pop
            pop = new_pop

            evaluation_start = time.time()

            fitness = fitness_resource_f.evaluate(pop)
            for ind, res_fit in zip(pop, fitness):
                ind.fitness.values = res_fit

            evaluation_time += time.time() - evaluation_start

            hof.update(pop)

            if best_fitness[0] <= deadline:
                # Optimizing resources
                plateau_steps = 0
                new_generation_number = generation_number - generation + 1
                max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number
                best_fitness = hof[0].fitness.values

                if len(pop) < population_size:
                    individuals_to_copy = rand.choices(pop, k=population_size - len(pop))
                    copied_individuals = [toolbox.copy_individual(ind) for ind in individuals_to_copy]
                    for copied_ind, ind in zip(copied_individuals, individuals_to_copy):
                        copied_ind.fitness.values = ind.fitness.values
                        copied_ind.time = ind.time
                    pop += copied_individuals

                while generation <= generation_number and plateau_steps < max_plateau_steps \
                        and (time_border is None or time.time() - global_start < time_border):
                    if verbose:
                        print(f'-- Generation {generation}, population={len(pop)}, best peak={best_fitness} --')

                    rand.shuffle(pop)

                    offspring_chromosomes = make_offspring(toolbox, pop, optimize_resources)
                    offspring = [toolbox.Individual(chromosome) for chromosome in offspring_chromosomes]

                    evaluation_start = time.time()

                    fitness = fitness_f.evaluate(offspring)

                    for ind, t in zip(offspring, fitness):
                        ind.time = t[0]

                    offspring = [ind for ind in offspring if ind.time <= deadline]

                    fitness_res = fitness_resource_f.evaluate(offspring)

                    for ind, res_fit in zip(offspring, fitness_res):
                        ind.fitness.values = res_fit

                    evaluation_time += time.time() - evaluation_start

                    # renewing population
                    pop += offspring
                    pop = toolbox.select(pop)
                    hof.update(pop)

                    prev_best_fitness = best_fitness
                    best_fitness = hof[0].fitness.values
                    plateau_steps = plateau_steps + 1 if best_fitness == prev_best_fitness else 0

                    generation += 1

        native.close()

        if verbose:
            print(f'Final fitness: {best_fitness}')
            print(f'Generations processing took {(time.time() - start) * 1000} ms')
            print(f'Full genetic processing took {(time.time() - global_start) * 1000} ms')
            print(f'Evaluation time: {evaluation_time * 1000}')

        best_chromosomes = [chromosome for chromosome in hof]

    best_schedules = [toolbox.chromosome_to_schedule(best_chromosome, landscape=landscape, timeline=timeline)
                      for best_chromosome in best_chromosomes]
    best_schedules = [({node.id: work for node, work in scheduled_works.items()},
                       schedule_start_time, timeline, order_nodes)
                      for scheduled_works, schedule_start_time, timeline, order_nodes in best_schedules]

    return best_schedules


def compare_individuals(first: ChromosomeType, second: ChromosomeType) -> bool:
    return (first[0] == second[0]).all() and (first[1] == second[1]).all() and (first[2] == second[2]).all()


def make_offspring(toolbox: Toolbox, population: list[ChromosomeType], optimize_resources: bool) \
        -> list[ChromosomeType]:
    offspring = []

    for ind1, ind2 in zip(population[::2], population[1::2]):
        # mate
        offspring.extend(toolbox.mate(ind1, ind2, optimize_resources))

    for mutant in offspring:
        # resource borders mutation
        toolbox.mutate_resource_borders(mutant)
        # other mutation
        toolbox.mutate(mutant)

    return offspring
