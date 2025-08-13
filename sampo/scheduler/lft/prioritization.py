from typing import Callable
from itertools import chain

import numpy as np
import random

from sampo.utilities.priority import extract_priority_groups_from_nodes
from sampo.schemas.graph import GraphNode


def map_lft_lst(head_nodes: list[GraphNode],
                node_id2child_ids: dict[str, set[str]],
                node_id2duration: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    project_max_duration = sum(node_id2duration.values())
    node_id2lft = {head_nodes[-1].id: project_max_duration}
    node_id2lst = {head_nodes[-1].id: project_max_duration - node_id2duration[head_nodes[-1].id]}

    for node in reversed(head_nodes[:-1]):
        suc_lst = [node_id2lst[child_id] for child_id in node_id2child_ids[node.id] if child_id in node_id2lst]
        node_id2lft[node.id] = min(suc_lst, default=0)
        node_id2lst[node.id] = node_id2lft[node.id] - node_id2duration[node.id]

    return node_id2lft, node_id2lst


def lft_prioritization(head_nodes: list[GraphNode],
                       node_id2parent_ids: dict[str, set[str]],
                       node_id2child_ids: dict[str, set[str]],
                       node_id2duration: dict[str, int],
                       core_f: Callable[[list[GraphNode],
                                         dict[str, set[str]],
                                         dict[str, set[str]],
                                         dict[str, int],
                                         list[list[GraphNode]],
                                         random.Random], list[GraphNode]],
                       rand: random.Random() = random.Random()) -> list[GraphNode]:
    """
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    """

    # form groups
    groups = extract_priority_groups_from_nodes(head_nodes)
    groups = [groups[priority] for priority in sorted(groups.keys())]

    result = core_f(head_nodes, node_id2parent_ids, node_id2child_ids, node_id2duration, groups, rand)

    return result


def lft_prioritization_core(head_nodes: list[GraphNode],
                            node_id2parent_ids: dict[str, set[str]],
                            node_id2child_ids: dict[str, set[str]],
                            node_id2duration: dict[str, int],
                            groups: list[list[GraphNode]],
                            rand: random.Random = random.Random()) -> list[GraphNode]:
    """
    Return nodes ordered by MIN-LFT priority rule.
    """
    node_id2lft, _ = map_lft_lst(head_nodes, node_id2child_ids, node_id2duration)

    ordered_groups_it = (sorted(group, key=lambda node: node_id2lft[node.id]) for group in groups)
    ordered_nodes = list(chain.from_iterable(ordered_groups_it))

    return ordered_nodes


def lft_randomized_prioritization_core(head_nodes: list[GraphNode],
                                       node_id2parent_ids: dict[str, set[str]],
                                       node_id2child_ids: dict[str, set[str]],
                                       node_id2duration: dict[str, int],
                                       groups: list[list[GraphNode]],
                                       rand: random.Random = random.Random()) -> list[GraphNode]:
    """
    Return ordered nodes sampled by MIN-LFT and MIN-LST priority rules.
    """
    node_id2group_priority = {node.id: i for i, group in enumerate(groups) for node in group}

    def is_eligible(node_id):
        return node_id2parent_ids[node_id].issubset(selected_ids_set)

    nodes2lft, nodes2lst = map_lft_lst(head_nodes, node_id2child_ids, node_id2duration)

    ordered_node_ids = []
    selected_ids_set = set()
    candidates = {head_nodes[0].id}

    while candidates:
        eligibles = [node_id for node_id in candidates if is_eligible(node_id)]
        min_group_priority = min(node_id2group_priority[node_id] for node_id in eligibles)
        eligibles = [node_id for node_id in eligibles if node_id2group_priority[node_id] == min_group_priority]

        priority_mapper = nodes2lft if rand.random() < 0.5 else nodes2lst

        weights = np.array([priority_mapper[node_id] for node_id in eligibles])
        weights = weights.max() - weights + 1
        weights = weights / weights.sum()

        selected_id = rand.choices(eligibles, weights=weights)[0]

        ordered_node_ids.append(selected_id)
        selected_ids_set.add(selected_id)
        candidates.remove(selected_id)
        candidates.update([child_id for child_id in node_id2child_ids[selected_id]])

    ordered_nodes = sorted(head_nodes, key=lambda node: ordered_node_ids.index(node.id))

    return ordered_nodes
