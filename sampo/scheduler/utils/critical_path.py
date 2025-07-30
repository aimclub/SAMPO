from operator import itemgetter

from sampo.scheduler.utils import get_head_nodes_with_connections_mappings, \
    get_head_nodes_with_connections_mappings_nodes
from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade
from sampo.schemas import GraphNode, WorkTimeEstimator


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


def critical_path(nodes: list[GraphNode], work_estimator: WorkTimeEstimator):
    weights = {node.id: -work_priority(node, calculate_working_time_cascade, work_estimator) for node in nodes}
    head_nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings_nodes(nodes)
    return critical_path_inner(head_nodes, weights, node_id2parent_ids)


def critical_path_inner(nodes: list[GraphNode],
                        weights: dict[str, float],
                        node_id2parent_ids: dict[str, set[str]]):
    cum_weights = ford_bellman(nodes, weights, node_id2parent_ids)
    node_dict = {node.id: node for node in nodes}

    critical_path = []

    start_node_id = min(cum_weights.items(), key=itemgetter(1))[0]
    node = node_dict[start_node_id]

    while node.children:
        critical_path.append(node)

        node = min(node.children, key=lambda x: cum_weights[x.id])

    return critical_path
