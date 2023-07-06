from random import Random
from typing import List, Dict, Tuple

import numpy as np
from deap.base import Toolbox
from pytest import fixture

from sampo.scheduler.genetic.converter import ChromosomeType, convert_schedule_to_chromosome
from sampo.scheduler.genetic.operators import init_toolbox
from sampo.schemas.contractor import Contractor, WorkerContractorPool, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, AbstractWorkEstimator
from sampo.utilities.collections_util import reverse_dictionary


def get_params(works_count: int) -> Tuple[int, float, float, int]:
    if works_count < 300:
        size_selection = 20
    else:
        size_selection = works_count // 15

    if works_count < 300:
        mutate_order = 0.006
    else:
        mutate_order = 2 / works_count

    if works_count < 300:
        mutate_resources = 0.06
    else:
        mutate_resources = 18 / works_count

    if works_count < 300:
        size_of_population = 80
    elif 1500 > works_count >= 300:
        size_of_population = 50
    else:
        size_of_population = works_count // 50
    return size_selection, mutate_order, mutate_resources, size_of_population


def create_toolbox(wg: WorkGraph,
                   contractors: List[Contractor],
                   worker_pool: WorkerContractorPool,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: Dict[str, Schedule],
                   rand: Random,
                   spec: ScheduleSpec = ScheduleSpec(),
                   work_estimator: WorkTimeEstimator = AbstractWorkEstimator,
                   landscape: LandscapeConfiguration = LandscapeConfiguration()) -> Tuple[Toolbox, np.ndarray]:
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

    init_chromosomes: Dict[str, ChromosomeType] = \
        {name: convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule)
         for name, schedule in init_schedules.items()}

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
                        index2node_list,
                        parents,
                        Time(0),
                        work_estimator), resources_border


@fixture
def setup_toolbox(setup_default_schedules) -> tuple:
    (setup_wg, setup_contractors, setup_landscape_many_holders), setup_default_schedules = setup_default_schedules
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)

    selection_size, mutate_order, mutate_resources, size_of_population = get_params(setup_wg.vertex_count)
    rand = Random(123)
    work_estimator: WorkTimeEstimator = AbstractWorkEstimator()

    return create_toolbox(setup_wg,
                          setup_contractors,
                          setup_worker_pool,
                          selection_size,
                          mutate_order,
                          mutate_resources,
                          setup_default_schedules,
                          rand,
                          work_estimator=work_estimator,
                          landscape=setup_landscape_many_holders), setup_wg, setup_contractors, setup_default_schedules, \
           setup_landscape_many_holders
