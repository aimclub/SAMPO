from operator import itemgetter

from sampo.scheduler.utils.critical_path import ford_bellman
from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade
from sampo.schemas.graph import GraphNode
from sampo.schemas.time_estimator import WorkTimeEstimator


def prioritization_nodes(nodes: list[GraphNode],
                         node_id2parent_ids: dict[str, set[str]],
                         work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """
    if len(nodes) == 1:
        return nodes

    # inverse weights
    weights = {node.id: -work_priority(node, calculate_working_time_cascade, work_estimator)
               for node in nodes}

    path_weights = ford_bellman(nodes, weights, node_id2parent_ids)

    ordered_nodes = sorted(nodes, key=lambda node: path_weights[node.id], reverse=True)

    return ordered_nodes


def prioritization(head_nodes: list[GraphNode],
                   node_id2parent_ids: dict[str, set[str]],
                   node_id2child_ids: dict[str, set[str]],
                   work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """

    def update_priority(node, priority_value, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            return

        visited.add(node)

        node.work_unit.priority = min(node.work_unit.priority, priority_value)

        for parent in getattr(node, "parents", []):
            update_priority(parent, priority_value, visited)

    # check priorities
    for node in head_nodes:
        for parent_node in node.parents:
            if node.work_unit.priority < parent_node.work_unit.priority:
                update_priority(parent_node, node.work_unit.priority)
                # parent_node.work_unit.priority = node.work_unit.priority

    # form groups
    groups = {priority: [] for priority in set(node.work_unit.priority for node in head_nodes)}
    for node in head_nodes:
        groups[node.work_unit.priority].append(node)

    result = []
    for _, group in sorted(groups.items(), key=itemgetter(0), reverse=True):
        result.extend(prioritization_nodes(group, node_id2parent_ids, work_estimator))

    return result
