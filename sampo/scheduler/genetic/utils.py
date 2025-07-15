import random

import numpy as np
from deap.base import Toolbox

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler.genetic.converter import convert_schedule_to_chromosome, ScheduleGenerationScheme
from sampo.scheduler.genetic.operators import init_toolbox
from sampo.scheduler.utils import get_worker_contractor_pool, get_head_nodes_with_connections_mappings
from sampo.schemas import WorkGraph, Contractor, Schedule, GraphNode, LandscapeConfiguration, WorkTimeEstimator, Time
from sampo.schemas.schedule_spec import ScheduleSpec


def init_chromosomes_f(wg: WorkGraph,
                       contractors: list[Contractor],
                       init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                       landscape: LandscapeConfiguration = LandscapeConfiguration()):
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

    return init_chromosomes


def prepare_optimized_data_structures(wg: WorkGraph,
                                      contractors: list[Contractor],
                                      landscape: LandscapeConfiguration):
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

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count

    return (worker_pool, index2node, index2zone, work_id2index, worker_name2index, index2contractor_obj,
            worker_pool_indices, contractor2index, contractor_borders, node_indices, parents, children,
            resources_border)


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
        worker_pool_indices, contractor2index, contractor_borders, node_indices, parents, children, \
        resources_border = prepare_optimized_data_structures(wg, contractors, landscape)

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
                        is_multiobjective)
