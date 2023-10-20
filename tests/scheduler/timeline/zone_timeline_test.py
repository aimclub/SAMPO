import numpy as np
from pytest import fixture

from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTBetweenScheduler, HEFTScheduler
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.graph import WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.requirements import ZoneReq
from sampo.schemas.zones import ZoneConfiguration


@fixture
def setup_zoned_wg(setup_rand, setup_simple_synthetic) -> WorkGraph:
    wg = setup_simple_synthetic.work_graph(mode=SyntheticGraphType.PARALLEL, top_border=100)

    for node in wg.nodes:
        node.work_unit.zone_reqs.append(ZoneReq(kind='zone1', required_status=setup_rand.randint(0, 2)))

    return wg


@fixture(params=[(costs_mode, start_status_mode) for start_status_mode in range(3) for costs_mode in range(2)],
         ids=[f'Costs mode: {costs_mode}, start status mode: {start_status_mode}' for start_status_mode in range(3) for costs_mode in range(2)])
def setup_landscape_config(request) -> LandscapeConfiguration:
    costs_mode, start_status_mode = request.param

    match costs_mode:
        case 0:
            time_costs = np.array([
                [0, 1000, 1000],
                [1000, 1000, 1000],
                [1000, 1000, 1000]
            ])
        case 1:
            time_costs = np.array([
                [0, 0, 0],
                [0, 1000, 1000],
                [0, 1000, 1000]
            ])
        case _:
            raise ValueError('Illegal costs mode')

    match start_status_mode:
        case 0:
            start_status = 0
        case 1:
            start_status = 1
        case 2:
            start_status = 2
        case _:
            raise ValueError('Illegal start status mode')

    zone_config = ZoneConfiguration(start_statuses={'zone1': start_status},
                                    time_costs=time_costs)
    return LandscapeConfiguration(zone_config=zone_config)


@fixture(params=[HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler(), GeneticScheduler(5)],
         ids=['HEFTScheduler', 'HEFTBetweenScheduler', 'TopologicalScheduler', 'GeneticScheduler'])
def setup_scheduler(request) -> Scheduler:
    return request.param


def test_zoned_scheduling(setup_zoned_wg, setup_landscape_config, setup_scheduler):
    contractors = [get_contractor_by_wg(setup_zoned_wg, scaler=1000)]
    schedule = setup_scheduler.schedule(wg=setup_zoned_wg, contractors=contractors, landscape=setup_landscape_config)
    print(schedule.execution_time)
