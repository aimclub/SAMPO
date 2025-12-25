import random
import time

from deap import tools
from deap.base import Toolbox

from sampo.api.genetic_api import Individual
from sampo.base import SAMPO
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, FitnessFunction, TimeFitness
from sampo.scheduler.genetic.utils import prepare_optimized_data_structures, get_only_new_fitness, get_clustered_pairs
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.utils.fitness_history import FitnessHistory


def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   selection_size: int = 50,
                   mutate_order: float = 0.05,
                   mutate_resources: float = 0.05,
                   mutate_zones: float = 0.05,
                   init_schedules: dict[
                       str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]] = {},
                   rand: random.Random = random.Random(),
                   spec: ScheduleSpec = ScheduleSpec(),
                   fitness_weights: tuple[int | float, ...] = (-1,),
                   work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                   sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                   assigned_parent_time: Time = Time(0),
                   landscape: LandscapeConfiguration = LandscapeConfiguration(),
                   only_lft_initialization: bool = False,
                   is_multiobjective: bool = False,
                   verbose: bool = True) -> Toolbox:
    start = time.time()

    worker_pool, index2node, index2zone, work_id2index, worker_name2index, index2contractor_obj, \
        worker_pool_indices, contractor2index, contractor_borders, node_indices, priorities, parents, children, \
        resources_border, contractors_available = prepare_optimized_data_structures(wg, contractors, landscape, spec)

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
                        priorities,
                        parents,
                        children,
                        resources_border,
                        contractors_available,
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
                    pop: list[ChromosomeType] = None,
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
                    is_multiobjective: bool = False,
                    offspring_types_list: list[str] | None = None,
                    eliminate_duplicates: bool = False,
                    save_history_to: str | None = None) \
        -> list[tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]]:
    return build_schedules_with_cache(wg, contractors, population_size, generation_number,
                                      mutpb_order, mutpb_res, mutpb_zones, init_schedules,
                                      rand, spec, weights, pop, landscape, fitness_object,
                                      fitness_weights, work_estimator, sgs_type, assigned_parent_time,
                                      timeline, time_border, max_plateau_steps, optimize_resources,
                                      deadline, only_lft_initialization, is_multiobjective,
                                      offspring_types_list, eliminate_duplicates, save_history_to)[0]


def build_schedules_with_cache(wg: WorkGraph,
                               contractors: list[Contractor],
                               population_size: int,
                               generation_number: int,
                               mutpb_order: float,
                               mutpb_res: float,
                               mutpb_zones: float,
                               init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                               rand: random.Random,
                               spec: ScheduleSpec,
                               weights: list[int] = None,
                               pop: list[ChromosomeType] = None,
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
                               is_multiobjective: bool = False,
                               offspring_types_list: list[str] | None = None,
                               eliminate_duplicates: bool = False,
                               save_history_to: str | None = None) \
        -> tuple[list[tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]], list[ChromosomeType]]:
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

    SAMPO.logger.info(f'Toolbox initialization & first population took {(time.time() - start) * 1000} ms')

    have_deadline = deadline is not None
    fitness_f = fitness_object if not have_deadline else TimeFitness()
    if have_deadline:
        toolbox.register_individual_constructor((-1,))

    # create population of a given size
    if pop is None:
        pop = SAMPO.backend.generate_first_population(population_size)
    else:
        pop = [toolbox.Individual(chromosome) for chromosome in pop]

    evaluation_start = time.time()

    hof = tools.ParetoFront(similar=compare_individuals)
    fitness_history = FitnessHistory()

    # map to each individual fitness function
    fitness = SAMPO.backend.compute_chromosomes(fitness_f, pop)

    evaluation_time = time.time() - evaluation_start

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    hof.update(pop)
    fitness_history.update(pop, hof, pop, comment="first generation")
    best_fitness = hof[0].fitness.values

    SAMPO.logger.info(f'First population evaluation took {evaluation_time * 1000} ms')

    start = time.time()

    generation = 1
    plateau_steps = 0
    new_generation_number = generation_number if not have_deadline else generation_number // 2
    new_max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number

    if offspring_types_list is None:
        offspring_types_list = generation_number * ["classical"]
    offspring_types_list = iter(offspring_types_list)


    while generation <= new_generation_number and plateau_steps < new_max_plateau_steps \
            and (time_border is None or time.time() - global_start < time_border):
        SAMPO.logger.info(f'-- Generation {generation}, population={len(pop)}, best fitness={best_fitness} --')
        current_offspring_type = next(offspring_types_list)

        rand.shuffle(pop)
        offspring = make_offspring(toolbox, pop, optimize_resources, rand, current_offspring_type)
        evaluation_start = time.time()
        offspring_fitness = SAMPO.backend.compute_chromosomes(fitness_f, offspring)
        for ind, fit in zip(offspring, offspring_fitness):
            ind.fitness.values = fit
        evaluation_time += time.time() - evaluation_start

        # renewing population
        if eliminate_duplicates:
            offspring = get_only_new_fitness(pop, offspring)
        pop += offspring
        pop = toolbox.select(pop)
        hof.update(pop)
        fitness_history.update(pop, hof, offspring, comment=current_offspring_type)

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
        fitness_history.update(pop, hof, pop, comment="first deadline population")

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

                offspring = make_offspring(toolbox, pop, optimize_resources, rand)

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
                fitness_history.update(pop, hof, offspring, comment="genetic deadline update")

                prev_best_fitness = best_fitness
                best_fitness = hof[0].fitness.values
                plateau_steps = plateau_steps + 1 if best_fitness == prev_best_fitness else 0

                generation += 1

    SAMPO.logger.info(f'Final fitness: {best_fitness}')
    SAMPO.logger.info(f'Generations processing took {(time.time() - start) * 1000} ms')
    SAMPO.logger.info(f'Full genetic processing took {(time.time() - global_start) * 1000} ms')
    SAMPO.logger.info(f'Evaluation time: {evaluation_time * 1000}')
    # save fitness history
    if save_history_to:
        fitness_history.save_to_json(path=save_history_to)

    best_chromosomes = [chromosome for chromosome in hof]

    best_schedules = [toolbox.chromosome_to_schedule(best_chromosome, landscape=landscape, timeline=timeline)
                      for best_chromosome in best_chromosomes]
    best_schedules = [({node.id: work for node, work in scheduled_works.items()},
                       schedule_start_time, timeline, order_nodes)
                      for scheduled_works, schedule_start_time, timeline, order_nodes in best_schedules]

    return best_schedules, pop


def make_offspring(toolbox: Toolbox, population: list[ChromosomeType], optimize_resources: bool, rand, offspring_type="classical") \
        -> list[Individual]:
    n_mutations = 1

    if offspring_type.startswith("classical"):
        only_swap_parts = int(offspring_type.split(":")[1]) == 1
        use_mate_resources_2 = int(offspring_type.split(":")[1]) == 2
        use_mate_resources_3 = int(offspring_type.split(":")[1]) == 3
        use_one_point_cross = int(offspring_type.split(":")[2]) == 1

        offspring = []
        for i1, i2 in zip(population[0::2], population[1::2]):
            offspring.extend(toolbox.mate(i1, i2, optimize_resources,
                only_swap_parts=only_swap_parts,
                use_mate_resources_2=use_mate_resources_2,
                use_mate_resources_3=use_mate_resources_3,
                use_one_point_cross=use_one_point_cross))

    elif offspring_type.startswith("clusters_crossover"):
        only_swap_parts = int(offspring_type.split(":")[1]) == 1
        use_mate_resources_2 = int(offspring_type.split(":")[1]) == 2
        use_mate_resources_3 = int(offspring_type.split(":")[1]) == 3
        use_one_point_cross = int(offspring_type.split(":")[2]) == 1
        n_clusters = int(offspring_type.split(":")[3])

        pairs = get_clustered_pairs([i.fitness.values for i in population], rand, n_clusters=n_clusters)
        copied_population = [toolbox.copy_individual(i) for i in population]  # copy, just in case

        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(
                copied_population[i1_index],
                copied_population[i2_index],
                optimize_resources,
                only_swap_parts=only_swap_parts,
                use_mate_resources_2=use_mate_resources_2,
                use_mate_resources_3=use_mate_resources_3,
                use_one_point_cross=use_one_point_cross
            ))


    elif offspring_type.startswith("only_mutations_half"):
        n_mutations = int(offspring_type.split(":")[1])
        better_half = toolbox.select(population, k=len(population)//2)
        offspring = [toolbox.copy_individual(i) for i in better_half] + [toolbox.copy_individual(i) for i in better_half]


    elif offspring_type.startswith("clustered_multi_1"):
        n_clusters = int(offspring_type.split(":")[1])
        groups = get_clustered_pairs([i.fitness.values for i in population], rand, n_clusters=n_clusters, return_groups=3)
        offspring = [
            toolbox.mate_multi_1(population[in1], population[in2], population[in3], optimize_resources=optimize_resources)
            for in1, in2, in3 in groups
        ]
    elif offspring_type.startswith("clustered_multi_2"):
        n_clusters = int(offspring_type.split(":")[1])
        groups = get_clustered_pairs([i.fitness.values for i in population], rand, n_clusters=n_clusters, return_groups=3)
        offspring = [
            toolbox.mate_multi_2(population[in1], population[in2], population[in3], optimize_resources=optimize_resources)
            for in1, in2, in3 in groups
        ]
    elif offspring_type.startswith("clustered_multi_3"):
        n_clusters = int(offspring_type.split(":")[1])
        groups = get_clustered_pairs([i.fitness.values for i in population], rand, n_clusters=n_clusters, return_groups=3)
        offspring = [
            toolbox.mate_multi_3(population[in1], population[in2], population[in3], optimize_resources=optimize_resources)
            for in1, in2, in3 in groups
        ]


    else:
        raise ValueError(f"Unknown offspring_type: {offspring_type}")


    for _ in range(n_mutations):
        # apply mutations
        for mutant in offspring:
            # main mutations
            toolbox.mutate(mutant)
            # resource borders mutation
            if optimize_resources:
                toolbox.mutate_resource_borders(mutant)

    return offspring


def compare_individuals(first: ChromosomeType, second: ChromosomeType) -> bool:
    """Decide if two individuals are 'similar enough' for the Pareto front"""
    same_fitness = first.fitness == second.fitness
    same_genomes = all(
        (first[i] == second[i]).all()
        for i in (0, 1, 2)
    )
    return same_fitness or same_genomes
