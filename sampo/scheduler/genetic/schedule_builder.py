import random
import time
import itertools

from deap import tools
from deap.base import Toolbox

from sampo.api.genetic_api import Individual
from sampo.base import SAMPO
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox, ChromosomeType, FitnessFunction, TimeFitness
from sampo.scheduler.genetic.utils import (
    prepare_optimized_data_structures,
    select_new_population_with_different_fitness,
    get_cluster_based_pairs,
    FitnessStats,
)
from sampo.scheduler.timeline.base import Timeline

from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


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
                    fitness_stats_path: str | None = None) \
        -> list[tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]]:
    return build_schedules_with_cache(wg, contractors, population_size, generation_number,
                                      mutpb_order, mutpb_res, mutpb_zones, init_schedules,
                                      rand, spec, weights, pop, landscape, fitness_object,
                                      fitness_weights, work_estimator, sgs_type, assigned_parent_time,
                                      timeline, time_border, max_plateau_steps, optimize_resources,
                                      deadline, only_lft_initialization, is_multiobjective, fitness_stats_path)[0]


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
                               fitness_stats_path: str | None = None) \
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

    fitness_stats = FitnessStats()
    hof = tools.ParetoFront(similar=compare_individuals)

    # map to each individual fitness function
    fitness = SAMPO.backend.compute_chromosomes(fitness_f, pop)

    evaluation_time = time.time() - evaluation_start

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    hof.update(pop); fitness_stats.update_history(pop, note="First Generation")
    best_fitness = hof[0].fitness.values

    SAMPO.logger.info(f'First population evaluation took {evaluation_time * 1000} ms')

    start = time.time()

    generation = 1
    plateau_steps = 0
    new_generation_number = generation_number if not have_deadline else generation_number // 2
    new_max_plateau_steps = max_plateau_steps if max_plateau_steps is not None else new_generation_number

    update_types = [
        {"offsprings_type": "classical", "change_order": True, "change_resources": False, "drop_fitness_duplicates": True, "note": "Classical Update +-"},
        {"offsprings_type": "only_mutations_elite", "change_order": True, "change_resources": True, "drop_fitness_duplicates": True, "note": "Only Mutations Elite ++"},
        {"offsprings_type": "crossover_inclusters_7", "change_order": True, "change_resources": False, "drop_fitness_duplicates": True, "note": "Crossover Clusters [7] +-"},
        {"offsprings_type": "only_mutations_elite", "change_order": False, "change_resources": True, "drop_fitness_duplicates": True, "note": "Only Mutations Elite -+"},
        {"offsprings_type": "crossover_inclusters_7", "change_order": True, "change_resources": True, "drop_fitness_duplicates": True, "note": "Crossover Clusters [7] ++"},
        {"offsprings_type": "only_mutations_good", "change_order": True, "change_resources": False, "drop_fitness_duplicates": True, "note": "Only Mutations Good +-"},
        {"offsprings_type": "crossover_inclusters_7", "change_order": False, "change_resources": True, "drop_fitness_duplicates": True, "note": "Crossover Clusters [7] -+"},
        {"offsprings_type": "only_mutations_elite_twice", "change_order": True, "change_resources": False, "drop_fitness_duplicates": True, "note": "Only Mutations Elite Twice +-"}
    ]
    # rand.shuffle(update_types)
    generation_update_params = itertools.cycle(update_types)

    while generation <= new_generation_number and plateau_steps < new_max_plateau_steps \
            and (time_border is None or time.time() - global_start < time_border):
        SAMPO.logger.info(f'-- Generation {generation}, population={len(pop)}, best fitness={best_fitness} --')

        current_generation_params = next(generation_update_params)

        # create offsprings
        rand.shuffle(pop)
        offspring = make_offspring(toolbox, pop, optimize_resources, rand,
            offsprings_type=current_generation_params["offsprings_type"],
            change_order=current_generation_params["change_order"],
            change_resources=current_generation_params["change_resources"]
        )
        # calculate fitness for offsprings
        evaluation_start = time.time()
        offspring_fitness = SAMPO.backend.compute_chromosomes(fitness_f, offspring)
        evaluation_time += time.time() - evaluation_start
        for ind, fit in zip(offspring, offspring_fitness):
            ind.fitness.values = fit

        # renewing population
        if current_generation_params["drop_fitness_duplicates"]:
            offspring = select_new_population_with_different_fitness(pop, offspring)
        pop += offspring
        pop = toolbox.select(pop)
        hof.update(pop); fitness_stats.update_history(pop, note=current_generation_params["note"])

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

        hof.update(pop); fitness_stats.update_history(pop, note="First Deadline Population")

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
                hof.update(pop); fitness_stats.update_history(pop, note="Genetic Deadline Update")

                prev_best_fitness = best_fitness
                best_fitness = hof[0].fitness.values
                plateau_steps = plateau_steps + 1 if best_fitness == prev_best_fitness else 0

                generation += 1

    SAMPO.logger.info(f'Final fitness: {best_fitness}')
    SAMPO.logger.info(f'Generations processing took {(time.time() - start) * 1000} ms')
    SAMPO.logger.info(f'Full genetic processing took {(time.time() - global_start) * 1000} ms')
    SAMPO.logger.info(f'Evaluation time: {evaluation_time * 1000}')
    fitness_stats.write_fitness_info(path=fitness_stats_path)

    best_chromosomes = [chromosome for chromosome in hof]

    best_schedules = [toolbox.chromosome_to_schedule(best_chromosome, landscape=landscape, timeline=timeline)
                      for best_chromosome in best_chromosomes]
    best_schedules = [({node.id: work for node, work in scheduled_works.items()},
                       schedule_start_time, timeline, order_nodes)
                      for scheduled_works, schedule_start_time, timeline, order_nodes in best_schedules]

    return best_schedules, pop


def compare_individuals(first: ChromosomeType, second: ChromosomeType) -> bool:
    return ((first[0] == second[0]).all() and (first[1] == second[1]).all() and (first[2] == second[2]).all()
            or first.fitness == second.fitness)


def make_offspring(toolbox: Toolbox, population: list[ChromosomeType], optimize_resources: bool, rand, offsprings_type="classical", change_order=True, change_resources=True) \
        -> list[Individual]:


    if offsprings_type == "classical":
        offspring = []
        for i1, i2 in zip(population[0::2], population[1::2]):
            offspring.extend(toolbox.mate(i1, i2, optimize_resources, change_order=change_order, change_resources=change_resources))

    # elif offsprings_type == "crossover_opposites":
    #     population_sorted = [
    #         toolbox.copy_individual(i)
    #         for i in sorted(population, key=lambda x: x.fitness.values)
    #     ]
    #     offspring = []
    #     for i1, i2 in zip(population_sorted, reversed(population_sorted)):
    #         offspring.extend(toolbox.mate(i1, i2, optimize_resources))

    # elif offsprings_type == "crossover_repeated":
    #     offspring = [toolbox.copy_individual(i) for i in population]
    #     for _ in range(2):
    #         rand.shuffle(offspring)
    #         next_offspring = []
    #         for i1, i2 in zip(offspring[0::2], offspring[1::2]):
    #             next_offspring.extend(toolbox.mate(i1, i2, optimize_resources))
    #         offspring = [toolbox.copy_individual(i) for i in next_offspring]

    # elif offsprings_type == "crossover_repeated_noreshuffle":
    #     offspring = [toolbox.copy_individual(i) for i in population]
    #     rand.shuffle(offspring)
    #     for _ in range(2):
    #         next_offspring = []
    #         for i1, i2 in zip(offspring[0::2], offspring[1::2]):
    #             next_offspring.extend(toolbox.mate(i1, i2, optimize_resources))
    #         offspring = [toolbox.copy_individual(i) for i in next_offspring]

    elif offsprings_type == "crossover_inclusters_3":
        pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand, n_clusters=3)
        population_copied = [toolbox.copy_individual(i) for i in population]
        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources, change_order=change_order, change_resources=change_resources))

    elif offsprings_type == "crossover_inclusters_5":
        pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand, n_clusters=5)
        population_copied = [toolbox.copy_individual(i) for i in population]
        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources, change_order=change_order, change_resources=change_resources))

    elif offsprings_type == "crossover_inclusters_7":
        pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand, n_clusters=7)
        population_copied = [toolbox.copy_individual(i) for i in population]
        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources, change_order=change_order, change_resources=change_resources))

    elif offsprings_type == "crossover_inclusters_8":
        pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand, n_clusters=8)
        population_copied = [toolbox.copy_individual(i) for i in population]
        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources, change_order=change_order, change_resources=change_resources))

    elif offsprings_type == "crossover_inclusters_9":
        pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand, n_clusters=9)
        population_copied = [toolbox.copy_individual(i) for i in population]
        offspring = []
        for i1_index, i2_index in pairs:
            offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources, change_order=change_order, change_resources=change_resources))

    # elif offsprings_type == "inclusters_repeated":
    #     pairs = get_cluster_based_pairs([i.fitness.values for i in population], rand)
    #     population_copied = [toolbox.copy_individual(i) for i in population]
    #     offspring = []
    #     for i1_index, i2_index in pairs:
    #         offspring.extend(toolbox.mate(population_copied[i1_index], population_copied[i2_index], optimize_resources))

    #     for _ in range(2):
    #         next_offspring = []
    #         for i1, i2 in zip(offspring[0::2], offspring[1::2]):
    #             next_offspring.extend(toolbox.mate(i1, i2, optimize_resources))
    #         offspring = [toolbox.copy_individual(i) for i in next_offspring]

    # elif offsprings_type == "elite_crossover":
    #     elite_population_oversample = [
    #         toolbox.copy_individual(i)
    #         for i in rand.choices(
    #             toolbox.select(population, k=len(population)//5),
    #             k=len(population)
    #         )
    #     ]
    #     offspring = []
    #     for i1, i2 in zip(population[0::2], elite_population_oversample[1::2]):
    #         offspring.extend(toolbox.mate(i1, i2, optimize_resources))

    # elif offsprings_type == "elite_elite_clusters":
    #     n_elite = len(population)//5
    #     if n_elite % 2 != 0:
    #         n_elite += 1

    #     elite_population = toolbox.select(population, k=n_elite)
    #     rand.shuffle(elite_population)

    #     offspring = []
    #     pairs = get_cluster_based_pairs([i.fitness.values for i in elite_population], rand, n_clusters=5)
    #     for i1_index, i2_index in pairs:
    #         i1 = toolbox.copy_individual(elite_population[i1_index])
    #         i2 = toolbox.copy_individual(elite_population[i2_index])
    #         offspring.extend(toolbox.mate(i1, i2, optimize_resources))

    elif offsprings_type in ("only_mutations", "only_mutations_twice", "only_mutations_thrice"):
        offspring = [toolbox.copy_individual(i) for i in population]

    elif offsprings_type in ("only_mutations_elite", "only_mutations_elite_twice"):
        offspring = [
            toolbox.copy_individual(i)
            for i in rand.choices(
                toolbox.select(population, k=len(population)//4),
                k=len(population)
            )
        ]

    elif offsprings_type in ("only_mutations_good", "only_mutations_good_twice"):
        offspring = [
            toolbox.copy_individual(i)
            for i in rand.choices(
                toolbox.select(population, k=len(population)//2),
                k=len(population)
            )
        ]

    # elif offsprings_type == "only_mutations_edges":
    #     n_elite = len(population)//2
    #     elite_population = toolbox.select(population, k=n_elite)
    #     selected = get_best_by_criteria_solutions([i.fitness.values for i in elite_population])
    #     offspring = [toolbox.copy_individual(elite_population[i]) for i in selected]


    # elif offsprings_type == "mutations_then_crossover":
    #     mutants = [toolbox.copy_individual(i) for i in population]

    #     # apply mutations
    #     for mutant in mutants:
    #         # main mutations
    #         toolbox.mutate(mutant)
    #         # resource borders mutation
    #         if optimize_resources:
    #             toolbox.mutate_resource_borders(mutant)

    #     rand.shuffle(mutants)
    #     offspring = []
    #     for i1, i2 in zip(mutants[0::2], mutants[1::2]):
    #         offspring.extend(toolbox.mate(i1, i2, optimize_resources))
    #     return offspring

    else:
        raise ValueError(f"Unknown offsprings_type: {offsprings_type}")


    n_mutation_applications = ({"only_mutations_twice": 2, "only_mutations_elite_twice": 2, "only_mutations_good_twice": 2}).get(offsprings_type, 1)
    for _ in range(n_mutation_applications):
        # apply mutations
        for mutant in offspring:
            # main mutations
            toolbox.mutate(mutant, change_order=change_order, change_resources=change_resources)
            # resource borders mutation
            if optimize_resources and change_resources:
                toolbox.mutate_resource_borders(mutant)

    return offspring
