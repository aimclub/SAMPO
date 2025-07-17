from operator import itemgetter

from sampo.scheduler.utils.priority import extract_priority_groups_from_nodes
from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade
from sampo.schemas.graph import GraphNode
from sampo.schemas.time_estimator import WorkTimeEstimator


def ford_bellman(nodes: list[GraphNode],
                 weights: dict[str, float],
                 node_id2parent_ids: dict[str, set[str]]) -> dict[str, float]:
    """
    Runs heuristic ford-bellman algorithm for given graph and weights.
    """
    path_weights: dict[str, float] = {node.id: 0 for node in nodes}
    # cache graph edges
    edges: list[tuple[str, str, float]] = sorted([(finish, start.id, weights[finish])
                                                  for start in nodes
                                                  for finish in node_id2parent_ids[start.id]
                                                  if finish in path_weights])
    # for changes heuristic
    changed = False
    # run standard ford-bellman on reversed edges
    # optimize dict access to finish weight
    for i in range(len(nodes)):
        cur_finish = edges[0][0]
        cur_finish_weight = path_weights[cur_finish]
        # relax on edges
        for finish, start, weight in edges:
            # we are running through the equality class by finish node
            # so if it changes renew the parameters of current equality class
            if cur_finish != finish:
                path_weights[cur_finish] = cur_finish_weight
                cur_finish = finish
                cur_finish_weight = path_weights[cur_finish]
            new_weight = path_weights[start] + weight
            if new_weight < cur_finish_weight:
                cur_finish_weight = new_weight
                changed = True
        # if we were done completely nothing with actual minimum weights, the algorithm ends
        if not changed:
            break
        # we go here if changed = True
        # so the last equality class weight can be changed, save it
        path_weights[cur_finish] = cur_finish_weight
        # next iteration should start without change info from previous
        changed = False

    return path_weights


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

    # form groups
    groups = extract_priority_groups_from_nodes(head_nodes)

    result = []
    for _, group in sorted(groups.items(), key=itemgetter(0), reverse=True):
        result.extend(prioritization_nodes(group, node_id2parent_ids, work_estimator))

    return result
