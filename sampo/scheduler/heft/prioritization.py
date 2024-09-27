from operator import attrgetter
from typing import Generator

from sampo.scheduler.utils.time_computaion import work_priority, calculate_working_time_cascade
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.stochastic_graph import StochasticGraph
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections_util import build_index
from sampo.utilities.linked_list import LinkedList
from sampo.utilities.nodes import insert_nodes_between


def ford_bellman(nodes: list[GraphNode], weights: dict[GraphNode, float]) -> dict[GraphNode, float]:
    """
    Runs heuristic ford-bellman algorithm for given graph and weights.
    """
    path_weights: dict[GraphNode, float] = {node: 0 for node in nodes}
    # cache graph edges
    edges: list[tuple[GraphNode, GraphNode, float]] = sorted([(finish, start, weights[finish])
                                                              for start in nodes
                                                              for finish in start.parents],
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


def prioritization(wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """
    return prioritization_nodes(wg.nodes, work_estimator)


def prioritization_nodes(nodes: list[GraphNode], work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """
    # inverse weights
    weights = {node: -work_priority(node, calculate_working_time_cascade, work_estimator)
               for node in nodes}

    path_weights = ford_bellman(nodes, weights)

    ordered_nodes = [i[0] for i in sorted(path_weights.items(), key=lambda x: (x[1], x[0].id))
                     if not i[0].is_inseparable_son()]

    return ordered_nodes


def stochastic_prioritization(wg: StochasticGraph, work_estimator: WorkTimeEstimator) -> Generator[GraphNode]:
    it = wg.iterate()
    first_node = next(it)
    seen_set = set()
    seen: dict[GraphNode, float] = {}
    pending_works = {first_node}

    # this algo seems to be quadratic, but Dijkstra-like algo
    # cannot be implemented because of maximization task
    while len(pending_works) > 0:
        for node in pending_works:
            if len(seen_set.intersection(node.parents_set)) == len(node.parents_set):
                # edges satisfied, adding to priority
                max_path_to_node = max([seen[v] for v in node.parents])
                node_cumulative_path = max_path_to_node + wg.average_labor_cost(node)
                seen[node] = node_cumulative_path
                seen_set.add(node)
                for child in node.children_set:
                    pending_works.add(child)

                yield node

                generated_graphs = wg.generate_next(node)

                for generated_nodes in generated_graphs:
                    inner_prioritization = prioritization_nodes(generated_nodes, work_estimator)
                    insert_nodes_between(generated_nodes, [node], node.children)

                    for v in inner_prioritization:
                        yield v
