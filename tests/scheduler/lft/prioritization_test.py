from sampo.scheduler.lft.prioritization import lft_prioritization
from sampo.scheduler.utils import get_worker_contractor_pool, get_head_nodes_with_connections_mappings
from sampo.schemas.graph import GraphNode
from tests.scheduler.lft.fixtures import setup_schedulers_and_parameters


def test_correct_order(setup_schedulers_and_parameters):
    setup_wg, setup_contractors, _, scheduler = setup_schedulers_and_parameters
    worker_pool = get_worker_contractor_pool(setup_contractors)
    nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings(setup_wg)
    node_id2duration = scheduler._contractor_workers_assignment(nodes, setup_contractors, worker_pool)
    order = lft_prioritization(nodes, node_id2parent_ids, node_id2child_ids, node_id2duration)

    seen: set[GraphNode] = set()
    for node in reversed(order):
        seen.update(node.get_inseparable_chain_with_self())
        for inode in node.get_inseparable_chain_with_self():
            assert all(pnode in seen for pnode in inode.parents)
