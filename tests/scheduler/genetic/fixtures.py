from random import Random
from typing import List, Dict, Tuple

import numpy as np
from deap.base import Toolbox
from pytest import fixture

from sampo.scheduler.genetic.converter import ChromosomeType, convert_schedule_to_chromosome
from sampo.scheduler.genetic.operators import init_toolbox
from sampo.scheduler.genetic.schedule_builder import create_toolbox
from sampo.schemas.contractor import Contractor, WorkerContractorPool, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
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


@fixture
def setup_toolbox(setup_default_schedules) -> tuple:
    (setup_wg, setup_contractors, setup_landscape_many_holders), setup_default_schedules = setup_default_schedules
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)

    selection_size, mutate_order, mutate_resources, size_of_population = get_params(setup_wg.vertex_count)
    rand = Random(123)
    work_estimator: WorkTimeEstimator = DefaultWorkEstimator()

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
