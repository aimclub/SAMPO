from operator import itemgetter

from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.time_estimator import WorkTimeEstimator


def ford_bellman(nodes: list[GraphNode], weights: dict[GraphNode, float]) -> dict[GraphNode, float]:
    """
    Runs heuristic ford-bellman algorithm for given graph and weights.
    """
    path_weights: dict[GraphNode, float] = {node: 0 for node in nodes}
    # cache graph edges
    edges: list[tuple[GraphNode, GraphNode, float]] = sorted([(finish, start, weights[finish])
                                                              for start in nodes
                                                              for finish in start.parents if finish in path_weights],
                                                             key=lambda x: (x[0].id, x[1].id))
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
            if cur_finish.id != finish.id:
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


def prioritization_nodes(nodes: list[GraphNode], work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """
    if len(nodes) == 1:
        return nodes

    # inverse weights
    weights = {node: -work_priority(node, calculate_working_time_cascade, work_estimator)
               for node in nodes}

    path_weights = ford_bellman(nodes, weights)

    ordered_nodes = [i[0] for i in sorted(path_weights.items(), key=lambda x: (x[1], x[0].id), reverse=True)
                     if not i[0].is_inseparable_son()]

    return ordered_nodes


def prioritization(wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """
    # form groups
    groups = {priority: [] for priority in set(node.work_unit.priority for node in wg.nodes)}
    for node in wg.nodes:
        groups[node.work_unit.priority].append(node)

    result = []
    for _, group in sorted(groups.items(), key=itemgetter(0), reverse=True):
        result.extend(prioritization_nodes(group, work_estimator))

    return result
