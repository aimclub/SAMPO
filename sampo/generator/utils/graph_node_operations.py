import queue

from sampo.schemas.graph import GraphNode


def count_node_ancestors(finish: GraphNode, root: GraphNode) -> int:
    """
    Counts the number of ancestors of the whole graph.

    :param finish: The node for which ancestors are to be counted.
    :param root: The root node of the graph.
    :return:
    """
    q = queue.Queue()
    count = 0
    used = set()
    used.add(root)
    q.put(finish)
    while not q.empty():
        node = q.get()
        for parent in node.parents:
            if parent in used:
                continue
            used.add(parent)
            q.put(parent)
            count += 1

    return count
