import numpy as np
import random

from sampo.schemas.graph import GraphNode


def map_lft_lst(head_nodes: list[GraphNode],
                node_id2child_ids: dict[str, set[str]],
                node_id2duration: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    project_max_duration = sum(node_id2duration.values())
    node_id2lft = {head_nodes[-1].id: project_max_duration}
    node_id2lst = {head_nodes[-1].id: project_max_duration - node_id2duration[head_nodes[-1].id]}

    for node in reversed(head_nodes[:-1]):
        suc_lst = [node_id2lst[child_id] for child_id in node_id2child_ids[node.id]]
        node_id2lft[node.id] = min(suc_lst)
        node_id2lst[node.id] = node_id2lft[node.id] - node_id2duration[node.id]

    return node_id2lft, node_id2lst


def lft_prioritization(head_nodes: list[GraphNode],
                       node_id2parent_ids: dict[str, set[str]],
                       node_id2child_ids: dict[str, set[str]],
                       node_id2duration: dict[str, int]) -> list[GraphNode]:
    """
    Return nodes ordered by MIN-LFT priority rule.
    """
    node_id2lft, _ = map_lft_lst(head_nodes, node_id2child_ids, node_id2duration)

    ordered_nodes = sorted(head_nodes, key=lambda node: node_id2lft[node.id], reverse=True)

    return ordered_nodes


def lft_randomized_prioritization(head_nodes: list[GraphNode],
                                  node_id2parent_ids: dict[str, set[str]],
                                  node_id2child_ids: dict[str, set[str]],
                                  node_id2duration: dict[str, int],
                                  rand: random.Random = random.Random()) -> list[GraphNode]:
    """
    Return ordered nodes sampled by MIN-LFT and MIN-LST priority rules.
    """

    def is_eligible(node_id):
        return node_id2parent_ids[node_id].issubset(selected_ids_set)

    nodes2lft, nodes2lst = map_lft_lst(head_nodes, node_id2child_ids, node_id2duration)

    ordered_node_ids = []
    selected_ids_set = set()
    candidates = {head_nodes[0].id}

    while candidates:
        eligibles = [node_id for node_id in candidates if is_eligible(node_id)]

        priority_mapper = nodes2lft if rand.random() < 0.5 else nodes2lst

        weights = np.array([priority_mapper[node_id] for node_id in eligibles])
        weights = weights.max() - weights + 1
        weights = weights / weights.sum()

        selected_id = rand.choices(eligibles, weights=weights)[0]

        ordered_node_ids.append(selected_id)
        selected_ids_set.add(selected_id)
        candidates.remove(selected_id)
        candidates.update([child_id for child_id in node_id2child_ids[selected_id]])

    ordered_nodes = sorted(head_nodes, key=lambda node: ordered_node_ids.index(node.id), reverse=True)

    return ordered_nodes
