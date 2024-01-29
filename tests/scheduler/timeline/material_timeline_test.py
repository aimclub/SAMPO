import uuid

import pytest
from _pytest.fixtures import fixture

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.material_timeline import SupplyTimeline
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import DefaultWorkEstimator


@fixture(scope='function')
def setup_timeline(setup_scheduler_parameters):
    return SupplyTimeline(landscape_config=setup_scheduler_parameters[-1])


def test_init_resource_structure(setup_timeline):
    timeline = setup_timeline

    assert len(timeline._timeline) != 0
    for res_holder_info in timeline._timeline.values():
        for state in res_holder_info.values():
            assert len(state) == 1


def test_supply_resources(setup_scheduler_parameters, setup_rand):
    wg, contractors, landscape = setup_scheduler_parameters
    if wg.vertex_count > 20:
        pytest.skip('Not manual graph')
    timeline = SupplyTimeline(landscape)

    ordered_nodes = prioritization(wg, DefaultWorkEstimator())
    for nd in ordered_nodes:
        if nd.work_unit.is_service_unit:
            continue
        nd.platform = landscape.platforms[setup_rand.randint(0, len(landscape.platforms) - 1)]

    parent_time = Time(0)
    for node in ordered_nodes[-1::-1]:
        materials = [Material(str(uuid.uuid4()), req.kind, req.count) for req in node.work_unit.material_reqs]
        delivery, parent_time = timeline._supply_resources(node,
                                                           parent_time,
                                                           materials, True)

    assert not parent_time.is_inf()
