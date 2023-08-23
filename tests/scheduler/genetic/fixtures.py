from random import Random
from typing import Tuple

from pytest import fixture

from sampo.schemas.graph import GraphNode
import numpy as np

from sampo.scheduler.genetic.schedule_builder import create_toolbox
from sampo.schemas.contractor import get_worker_contractor_pool
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


def get_params(works_count: int) -> Tuple[float, float, int]:
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
    return mutate_order, mutate_resources, size_of_population


@fixture
def setup_toolbox(setup_default_schedules) -> tuple:
    (setup_wg, setup_contractors, setup_landscape_many_holders), setup_default_schedules = setup_default_schedules
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)

    mutate_order, mutate_resources, size_of_population = get_params(setup_wg.vertex_count)
    rand = Random(123)
    work_estimator: WorkTimeEstimator = DefaultWorkEstimator()

    nodes = [node for node in setup_wg.nodes if not node.is_inseparable_son()]
    worker_name2index = {worker_name: index for index, worker_name in enumerate(setup_worker_pool)}
    resources_min_border = np.zeros(len(setup_worker_pool))
    for work_index, node in enumerate(nodes):
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_min_border[worker_index] = max(resources_min_border[worker_index], req.min_count)

    return (create_toolbox(setup_wg,
                           setup_contractors,
                           setup_worker_pool,
                           size_of_population,
                           mutate_order,
                           mutate_resources,
                           setup_default_schedules,
                           rand,
                           work_estimator=work_estimator,
                           landscape=setup_landscape_many_holders), resources_min_border,
            setup_wg, setup_contractors, setup_default_schedules, setup_landscape_many_holders)
