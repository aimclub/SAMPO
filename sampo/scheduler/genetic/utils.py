import random
from operator import attrgetter

import numpy as np
from deap.base import Toolbox

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox
from sampo.scheduler.utils import get_worker_contractor_pool, get_head_nodes_with_connections_mappings
from sampo.schemas import WorkGraph, Contractor, Schedule, GraphNode, LandscapeConfiguration, WorkTimeEstimator, Time
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.base import SAMPO

def init_chromosomes_f(wg: WorkGraph,
                       contractors: list[Contractor],
                       spec: ScheduleSpec,
                       init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                       landscape: LandscapeConfiguration = LandscapeConfiguration()):
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

    return init_chromosomes


def prepare_optimized_data_structures(wg: WorkGraph,
                                      contractors: list[Contractor],
                                      landscape: LandscapeConfiguration,
                                      spec: ScheduleSpec):
    # preparing access-optimized data structures
    index2zone = {ind: zone for ind, zone in enumerate(landscape.zone_config.start_statuses)}

    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = {contractor.id: ind for ind, contractor in enumerate(contractors)}

    worker_pool = get_worker_contractor_pool(contractors)
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}

    contractor_borders = np.zeros((len(contractor2index), len(worker_name2index)), dtype=int)
    for ind, contractor in enumerate(contractors):
        for ind_worker, worker in enumerate(contractor.workers.values()):
            contractor_borders[ind, ind_worker] = worker.count

    nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings(wg)
    node_indices = list(range(len(nodes)))

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    children = {work_id2index[node_id]: set(work_id2index[child_id] for child_id in child_ids)
                for node_id, child_ids in node_id2child_ids.items()}
    parents = {work_id2index[node_id]: set(work_id2index[parent_id] for parent_id in parent_ids)
               for node_id, parent_ids in node_id2parent_ids.items()}

    priorities = np.array([index2node[i].work_unit.priority for i in range(len(index2node))])

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    contractors_available = np.zeros((len(index2node), len(contractor2index)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count

        contractors_spec = spec.get_work_spec(node.id).contractors or map(attrgetter('id'), contractors)
        for contractor in contractors_spec:
            contractors_available[work_index, contractor2index[contractor]] = 1

    return (worker_pool, index2node, index2zone, work_id2index, worker_name2index, index2contractor_obj,
            worker_pool_indices, contractor2index, contractor_borders, node_indices, priorities, parents,
            children, resources_border, contractors_available)


def create_toolbox_using_cached_chromosomes(wg: WorkGraph,
                                            contractors: list[Contractor],
                                            population_size: int,
                                            mutate_order: float,
                                            mutate_resources: float,
                                            mutate_zones: float,
                                            init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                                            rand: random.Random,
                                            spec: ScheduleSpec = ScheduleSpec(),
                                            work_estimator: WorkTimeEstimator = None,
                                            assigned_parent_time: Time = Time(0),
                                            fitness_weights: tuple[int | float, ...] = (-1,),
                                            landscape: LandscapeConfiguration = LandscapeConfiguration(),
                                            sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                                            only_lft_initialization: bool = False,
                                            is_multiobjective: bool = False) -> Toolbox:
    worker_pool, index2node, index2zone, work_id2index, worker_name2index, index2contractor_obj, \
        worker_pool_indices, contractor2index, contractor_borders, node_indices, priorities, parents, children, \
        resources_border, contractors_available = prepare_optimized_data_structures(wg, contractors, landscape, spec)

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


class FitnessStats:
    """
    Class to track fitness and other stats for genetic algorithm
    [experimental]
    """

    def __init__(self):
        # [generation1, generation2, ...]
        self.fitness_history = []
        # comments about how this generation was created, optional
        self.notes = []

    def update_history(self, population, note=""):
        fitness_values = [i.fitness.values for i in population]
        self.fitness_history.append(fitness_values)
        self.notes.append(note)

    def get_uniqueness_scores(self):
        """
        Calculate uniqueness of fitness values in population
        How many genomes with the same fitness are in the population
        range: (0, 1], more = more fitness values are unique
        """
        uniqueness_scores = [
            len(set(fitness_values)) / len(fitness_values)
            for fitness_values in self.fitness_history
        ]
        # round values for readability
        uniqueness_scores = [round(score, 4) for score in uniqueness_scores]
        return uniqueness_scores


    def get_generation_shifts(self):
        """
        Calculate shift of fitness values in new generation
        How much population changes after update
        range: [0, 1], more = more changes in population
        """
        generation_shifts = [0]  # to match the len of list and number of generations
        n_generations = len(self.fitness_history)
        for i in range(1, n_generations):
            old_fitness = set(self.fitness_history[i-1])
            new_fitness = set(self.fitness_history[i])

            shift_score = len( new_fitness.difference(old_fitness) ) / len(new_fitness)
            generation_shifts.append(shift_score)

        # round values for readability
        generation_shifts = [round(score, 4) for score in generation_shifts]
        return generation_shifts

    def log_fitness_info(self):
        try:
            uniqueness_scores = self.get_uniqueness_scores()
            generation_shifts = self.get_generation_shifts()
            SAMPO.logger.info(f'"fitness_uniqueness": {uniqueness_scores}, "generations_shifts": {generation_shifts}, "notes": {self.notes}')

        except Exception as ex:  # just in case
            SAMPO.logger.info(f"Error occured when trying to log fitness info: {str(ex)}")

