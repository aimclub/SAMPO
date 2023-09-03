import numpy as np
from pytest import fixture

from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.heft.base import HEFTBetweenScheduler
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

@fixture
def setup_landscape_config() -> LandscapeConfiguration:
    zone_config = ZoneConfiguration(start_statuses={'zone1': 1},
                                    time_costs=np.array([
                                        [0, 0, 0],
                                        [0, 1, 1],
                                        [0, 1, 1]
                                    ]))
    return LandscapeConfiguration(zone_config=zone_config)

def test_zoned_scheduling(setup_zoned_wg, setup_landscape_config):
    contractors = [get_contractor_by_wg(setup_zoned_wg)]
    scheduler = HEFTBetweenScheduler()
    schedule = scheduler.schedule(setup_zoned_wg, contractors, landscape=setup_landscape_config)
    print()
