from typing import Set

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.utils import get_head_nodes_with_connections_mappings
from sampo.schemas.graph import GraphNode
from sampo.schemas.time_estimator import DefaultWorkEstimator


def test_correct_order(setup_wg):
    nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings(setup_wg)
    order = prioritization(nodes, node_id2parent_ids, node_id2child_ids, DefaultWorkEstimator())

    seen: Set[GraphNode] = set()
    for node in reversed(order):
        seen.update(node.get_inseparable_chain_with_self())
        for inode in node.get_inseparable_chain_with_self():
            assert all(pnode in seen for pnode in inode.parents)

