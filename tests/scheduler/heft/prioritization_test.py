from typing import Set

from sampo.scheduler.heft.prioritization import prioritization
from sampo.schemas.graph import GraphNode


def test_correct_order(setup_wg):
    order = prioritization(setup_wg)
    seen: Set[GraphNode] = set()

    for node in reversed(order):
        if not node.children:
            break
        assert all([parent in seen for parent in node.parents])
        seen.add(node)
        while node.is_inseparable_parent():
            node = node.inseparable_son
            seen.add(node)

