from sampo.scheduler.lft.prioritization import lft_prioritization
from sampo.schemas.graph import GraphNode
from sampo.schemas.contractor import get_worker_contractor_pool
from tests.scheduler.lft.fixtures import setup_schedulers_and_parameters


def test_correct_order(setup_schedulers_and_parameters):
    setup_wg, setup_contractors, _, scheduler = setup_schedulers_and_parameters
    worker_pool = get_worker_contractor_pool(setup_contractors)
    node_id2duration = scheduler._contractor_workers_assignment(setup_wg, setup_contractors, worker_pool)
    order = lft_prioritization(setup_wg, node_id2duration)
    seen: set[GraphNode] = set()

    for node in reversed(order):
        if not node.children:
            break
        assert all([parent in seen for parent in node.parents])
        seen.add(node)
        while node.is_inseparable_parent():
            node = node.inseparable_son
            seen.add(node)
