from sampo.schemas.graph import GraphNode
from collections import deque


def count_ancestors(first_ancestors: list[GraphNode], root: GraphNode) -> int:
    """
    Counts the number of ancestors of the whole graph.

    :param first_ancestors: First ancestors of ancestors which must be counted.
    :param root: The root node of the graph.
    :return:
    """
    q = deque(first_ancestors)
    count = len(first_ancestors)
    used = set()
    used.add(root)
    while q:
        node = q.pop()
        for parent in node.parents:
            if parent in used:
                continue
            used.add(parent)
            q.appendleft(parent)
            count += 1

    return count
