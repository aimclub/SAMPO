from operator import itemgetter

from sampo.scheduler.utils.critical_path import ford_bellman
from sampo.utilities.priority import extract_priority_groups_from_nodes
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

    ordered_nodes = sorted(nodes, key=lambda node: path_weights[node.id])

    return ordered_nodes


def prioritization(head_nodes: list[GraphNode],
                   node_id2parent_ids: dict[str, set[str]],
                   node_id2child_ids: dict[str, set[str]],
                   work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """

    # form groups
    groups = extract_priority_groups_from_nodes(head_nodes)

    result = []
    for _, group in sorted(groups.items(), key=itemgetter(0)):
        result.extend(prioritization_nodes(group, node_id2parent_ids, work_estimator))

    return result
