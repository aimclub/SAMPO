from functools import partial
from operator import itemgetter

from sampo.scheduler.utils import get_head_nodes_with_connections_mappings_nodes
from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade, \
    calculate_scheduled_time_cascade
from sampo.schemas import GraphNode, WorkTimeEstimator, ScheduledWork


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

    if not edges:
        return path_weights

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


def critical_path_graph(nodes: list[GraphNode], work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    weights = {node.id: -work_priority(node, calculate_working_time_cascade, work_estimator) for node in nodes}
    head_nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings_nodes(nodes)
    return _critical_path(head_nodes, weights, node_id2parent_ids, node_id2child_ids)


def critical_path_schedule(nodes: list[GraphNode], scheduled_works: dict[str, ScheduledWork]) -> list[GraphNode]:
    weights = {node.id: -work_priority(node, partial(calculate_scheduled_time_cascade, id2swork=scheduled_works), None) for node in nodes}
    head_nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings_nodes(nodes)
    return _critical_path(head_nodes, weights, node_id2parent_ids, node_id2child_ids)


def critical_path_schedule_lag_optimized(nodes_optimized: list[GraphNode],
                                         scheduled_works_optimized: dict[str, ScheduledWork],
                                         nodes: list[GraphNode]) -> list[GraphNode]:
    node_dict = {node.id: node for node in nodes}

    cp_optimized = critical_path_schedule(nodes_optimized, scheduled_works_optimized)
    cp = []
    for node in cp_optimized:
        while node.is_inseparable_parent():
            node = node.inseparable_son
        if node_dict[node.id] not in cp:
            cp.append(node_dict[node.id])
    return cp


def _critical_path(nodes: list[GraphNode],
                   weights: dict[str, float],
                   node_id2parent_ids: dict[str, set[str]],
                   node_id2child_ids: dict[str, set[str]]) -> list[GraphNode]:
    cum_weights = ford_bellman(nodes, weights, node_id2parent_ids)
    node_dict = {node.id: node for node in nodes}

    critical_path = []

    start_node_id = min(cum_weights.items(), key=itemgetter(1))[0]
    node = node_dict[start_node_id]

    while node.children:
        critical_path.append(node)

        # sorted_nodes = sorted(node.children, key=lambda x: cum_weights[x.id])
        # [(node, cum_weights[node.id]) for node in critical_path]
        # node = sorted_nodes[0]
        node = node_dict[min(node_id2child_ids[node.id], key=lambda x: cum_weights[x])]

    return critical_path
