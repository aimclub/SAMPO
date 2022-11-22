from operator import attrgetter
from random import Random
from typing import List, Dict, Tuple, Optional

import numpy as np
from deap.base import Toolbox
from pytest import fixture

from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.scheduler.genetic.converter import ChromosomeType, convert_schedule_to_chromosome
from sampo.scheduler import init_toolbox
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.utilities.collections import reverse_dictionary


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
                   agents: WorkerContractorPool,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: Dict[str, Schedule],
                   start: str,
                   rand: Random,
                   work_estimator: WorkTimeEstimator = None) -> Tuple[Toolbox, np.ndarray]:
    index2node: Dict[int, GraphNode] = {index: node for index, node in enumerate(wg.nodes)}
    work_id2index: Dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(agents)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)

    init_chromosomes: Dict[str, ChromosomeType] = \
        {name: convert_schedule_to_chromosome(index2node, work_id2index, worker_name2index,
                                              contractor2index, schedule)
         for name, schedule in init_schedules.items()}

    resources_border = np.zeros((2, len(agents), len(index2node)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = \
                min(req.max_count, max(list(map(attrgetter('count'), agents[req.kind].values()))))

    return init_toolbox(wg,
                        contractors,
                        agents,
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
                        work_estimator), resources_border


@fixture(scope='function')
def setup_toolbox(setup_wg, setup_contractors, setup_agents,
                  setup_start_date, setup_default_schedules) -> Tuple[Toolbox, np.ndarray]:
    selection_size, mutate_order, mutate_resources, size_of_population = get_params(setup_wg.vertex_count)
    rand = Random(123)
    work_estimator: Optional[WorkTimeEstimator] = None

    return create_toolbox(setup_wg,
                          setup_contractors,
                          setup_agents,
                          selection_size,
                          mutate_order,
                          mutate_resources,
                          setup_default_schedules,
                          setup_start_date,
                          rand,
                          work_estimator)
