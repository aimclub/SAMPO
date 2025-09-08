from random import Random

import numpy as np
from pytest import fixture

from sampo.scheduler.genetic.schedule_builder import create_toolbox
from sampo.scheduler.utils import get_worker_contractor_pool, get_head_nodes_with_connections_mappings
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


def get_params(works_count: int) -> tuple[float, float, float, int]:
    """
    Return base parameters for model to make new population

    :param works_count:
    :return:
    """
    mutate_order = 0.05
    mutate_resources = 0.005
    mutate_zones = 0.05

    if works_count < 300:
        size_of_population = 50
    elif 1500 > works_count >= 300:
        size_of_population = 100
    else:
        size_of_population = works_count // 25
    return mutate_order, mutate_resources, mutate_zones, size_of_population


@fixture
def setup_toolbox(setup_default_schedules) -> tuple:
    (wg, contractors, landscape, spec, rand), setup_default_schedules = setup_default_schedules
    setup_worker_pool = get_worker_contractor_pool(contractors)

    mutate_order, mutate_resources, mutate_zones, size_of_population = get_params(wg.vertex_count)
    work_estimator: WorkTimeEstimator = DefaultWorkEstimator()

    nodes, *_ = get_head_nodes_with_connections_mappings(wg)
    worker_name2index = {worker_name: index for index, worker_name in enumerate(setup_worker_pool)}
    resources_border = np.zeros((2, len(setup_worker_pool), len(nodes)))
    for work_index, node in enumerate(nodes):
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count

    return (create_toolbox(wg,
                           contractors,
                           size_of_population,
                           mutate_order,
                           mutate_resources,
                           mutate_zones,
                           setup_default_schedules,
                           rand,
                           spec=spec,
                           work_estimator=work_estimator,
                           landscape=landscape,
                           verbose=False), resources_border,
            wg, contractors, spec, rand, setup_default_schedules, landscape)
