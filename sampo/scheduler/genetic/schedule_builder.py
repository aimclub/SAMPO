import random
import time
from typing import Callable

from deap import tools
from deap.base import Toolbox

from sampo.api.genetic_api import Individual
from sampo.base import SAMPO
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, FitnessFunction, TimeFitness
from sampo.scheduler.genetic.utils import prepare_optimized_data_structures
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   mutate_zones: float,
                   init_schedules: dict[
                       str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                   rand: random.Random,
                   spec: ScheduleSpec = ScheduleSpec(),
                   fitness_weights: tuple[int | float, ...] = (-1,),
                   work_estimator: WorkTimeEstimator = None,
                   sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                   assigned_parent_time: Time = Time(0),
                   landscape: LandscapeConfiguration = LandscapeConfiguration(),
                   only_lft_initialization: bool = False,
                   is_multiobjective: bool = False,
                   verbose: bool = True) -> Toolbox:
    start = time.time()

    worker_pool, index2node, index2zone, work_id2index, worker_name2index, index2contractor_obj, \
        worker_pool_indices, contractor2index, contractor_borders, node_indices, parents, children, \
        resources_border = prepare_optimized_data_structures(wg, contractors, landscape)

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
                        selection_size,
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
                        is_multiobjective)


def build_schedules(wg: WorkGraph,
                    contractors: list[Contractor],
                    population_size: int,
                    generation_number: int,
                    mutpb_order: float,
                    mutpb_res: float,
                    mutpb_zones: float,
                    init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                    rand: random.Random,
                    spec: ScheduleSpec,
                    weights: list[int],
                    landscape: LandscapeConfiguration = LandscapeConfiguration(),
                    fitness_object: FitnessFunction = TimeFitness(),
                    fitness_weights: tuple[int | float, ...] = (-1,),
                    work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                    sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                    assigned_parent_time: Time = Time(0),
                    timeline: Timeline | None = None,
                    time_border: int | None = None,
                    max_plateau_steps: int | None = None,
                    optimize_resources: bool = False,
                    deadline: Time | None = None,
                    only_lft_initialization: bool = False,
                    is_multiobjective: bool = False) \
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

    toolbox = create_toolbox(wg, contractors, population_size,
                             mutpb_order, mutpb_res, mutpb_zones, init_schedules,
                             rand, spec, fitness_weights, work_estimator,
                             sgs_type, assigned_parent_time, landscape,
                             only_lft_initialization, is_multiobjective)

    SAMPO.backend.cache_scheduler_info(wg, contractors, landscape, spec, rand, work_estimator)
    SAMPO.backend.cache_genetic_info(population_size,
                                     mutpb_order, mutpb_res, mutpb_zones,
                                     deadline, weights,
                                     init_schedules, assigned_parent_time, fitness_weights,
                                     sgs_type, only_lft_initialization, is_multiobjective)

    # create population of a given size
    pop = SAMPO.backend.generate_first_population(population_size)

    SAMPO.logger.info(f'Toolbox initialization & first population took {(time.time() - start) * 1000} ms')

    have_deadline = deadline is not None
    fitness_f = fitness_object if not have_deadline else TimeFitness()
    if have_deadline:
        toolbox.register_individual_constructor((-1,))
    evaluation_start = time.time()

    hof = tools.ParetoFront(similar=compare_individuals)

    # map to each individual fitness function
    fitness = SAMPO.backend.compute_chromosomes(fitness_f, pop)

    evaluation_time = time.time() - evaluation_start

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    hof.update(pop)
    best_fitness = hof[0].fitness.values

    SAMPO.logger.info(f'First population evaluation took {evaluation_time * 1000} ms')

    start = time.time()

    generation = 1
    plateau_steps = 0
    new_generation_number = generation_number if not have_deadline else generation_number // 2
    new_max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number

    while generation <= new_generation_number and plateau_steps < new_max_plateau_steps \
            and (time_border is None or time.time() - global_start < time_border):
        SAMPO.logger.info(f'-- Generation {generation}, population={len(pop)}, best fitness={best_fitness} --')

        rand.shuffle(pop)

        offspring = make_offspring(toolbox, pop, optimize_resources)

        evaluation_start = time.time()

        offspring_fitness = SAMPO.backend.compute_chromosomes(fitness_f, offspring)

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
        fitness_resource_f = fitness_object
        toolbox.register_individual_constructor(fitness_weights)

        SAMPO.logger.info(f'Deadline not reached !!! Deadline {deadline} < best time {best_fitness}')
        # clear best individuals
        hof.clear()

        for ind in pop:
            ind.time = ind.fitness.values[0]

        if best_fitness[0] > deadline:
            SAMPO.logger.info(f'Deadline not reached !!! Deadline {deadline} < best time {best_fitness[0]}')
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

        fitness = SAMPO.backend.compute_chromosomes(fitness_resource_f, pop)
        for ind, res_fit in zip(pop, fitness):
            ind.fitness.values = res_fit

        evaluation_time += time.time() - evaluation_start

        hof.update(pop)

        if best_fitness[0] <= deadline:
            # Optimizing resources
            plateau_steps = 0
            new_generation_number = generation_number - generation + 1
            new_max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number
            best_fitness = hof[0].fitness.values

            if len(pop) < population_size:
                individuals_to_copy = rand.choices(pop, k=population_size - len(pop))
                copied_individuals = [toolbox.copy_individual(ind) for ind in individuals_to_copy]
                for copied_ind, ind in zip(copied_individuals, individuals_to_copy):
                    copied_ind.fitness.values = ind.fitness.values
                    copied_ind.time = ind.time
                pop += copied_individuals

            while generation <= generation_number and plateau_steps < new_max_plateau_steps \
                    and (time_border is None or time.time() - global_start < time_border):
                SAMPO.logger.info(f'-- Generation {generation}, population={len(pop)}, best peak={best_fitness} --')

                rand.shuffle(pop)

                offspring = make_offspring(toolbox, pop, optimize_resources)

                evaluation_start = time.time()

                fitness = SAMPO.backend.compute_chromosomes(fitness_f, offspring)

                for ind, t in zip(offspring, fitness):
                    ind.time = t[0]

                offspring = [ind for ind in offspring if ind.time <= deadline]

                fitness_res = SAMPO.backend.compute_chromosomes(fitness_resource_f, offspring)

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

    SAMPO.logger.info(f'Final fitness: {best_fitness}')
    SAMPO.logger.info(f'Generations processing took {(time.time() - start) * 1000} ms')
    SAMPO.logger.info(f'Full genetic processing took {(time.time() - global_start) * 1000} ms')
    SAMPO.logger.info(f'Evaluation time: {evaluation_time * 1000}')

    best_chromosomes = [chromosome for chromosome in hof]

    best_schedules = [toolbox.chromosome_to_schedule(best_chromosome, landscape=landscape, timeline=timeline)
                      for best_chromosome in best_chromosomes]
    best_schedules = [({node.id: work for node, work in scheduled_works.items()},
                       schedule_start_time, timeline, order_nodes)
                      for scheduled_works, schedule_start_time, timeline, order_nodes in best_schedules]

    return best_schedules


def compare_individuals(first: ChromosomeType, second: ChromosomeType) -> bool:
    return ((first[0] == second[0]).all() and (first[1] == second[1]).all() and (first[2] == second[2]).all()
            or first.fitness == second.fitness)


def make_offspring(toolbox: Toolbox, population: list[ChromosomeType], optimize_resources: bool) \
        -> list[Individual]:
    offspring = []

    for ind1, ind2 in zip(population[::2], population[1::2]):
        # mate
        offspring.extend(toolbox.mate(ind1, ind2, optimize_resources))

    for mutant in offspring:
        if optimize_resources:
            # resource borders mutation
            toolbox.mutate_resource_borders(mutant)
        # other mutation
        toolbox.mutate(mutant)

    return offspring
