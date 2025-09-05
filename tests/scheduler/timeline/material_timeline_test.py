import pytest
from _pytest.fixtures import fixture

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.hybrid_supply_timeline import HybridSupplyTimeline
from sampo.scheduler.utils import get_head_nodes_with_connections_mappings
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import DefaultWorkEstimator


@fixture(scope='function')
def setup_timeline(setup_scheduler_parameters):
    return HybridSupplyTimeline(landscape_config=setup_scheduler_parameters[-1])


def test_supply_resources(setup_scheduler_parameters, setup_rand):
    wg, contractors, landscape, _, _ = setup_scheduler_parameters
    if not landscape.platforms:
        pytest.skip('Non landscape test')
    timeline = HybridSupplyTimeline(landscape)

    nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings(wg)
    ordered_nodes = prioritization(nodes, node_id2parent_ids, node_id2child_ids, DefaultWorkEstimator())
    for node in ordered_nodes:
        if node.work_unit.is_service_unit:
            continue
        node.platform = landscape.platforms[setup_rand.randint(0, len(landscape.platforms) - 1)]

    delivery_time = Time(0)
    for node in ordered_nodes[-1::-1]:
        if node.work_unit.is_service_unit:
            continue
        materials = node.work_unit.need_materials()
        deadline = delivery_time
        delivery, delivery_time = timeline.deliver_resources(node,
                                                             deadline,
                                                             materials)
        assert delivery_time >= deadline

    assert not delivery_time.is_inf()
