from sampo.schemas.graph import GraphNode, WorkGraph
import numpy as np
import random


def map_lft_lst(wg: WorkGraph, node_id2duration: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    project_max_duration = sum(node_id2duration.values())
    nodes2lft = {wg.finish.id: project_max_duration}
    nodes2lst = {wg.finish.id: project_max_duration - node_id2duration[wg.finish.id]}

    for node in reversed(wg.nodes[:-1]):
        suc_lst = [nodes2lst[child.id] for child in node.children]
        nodes2lft[node.id] = min(suc_lst)
        nodes2lst[node.id] = nodes2lft[node.id] - node_id2duration[node.id]

    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    nodes2lft = {node.id: nodes2lft[node.get_inseparable_chain_with_self()[-1].id] for node in nodes}
    nodes2lst = {node.id: nodes2lst[node.get_inseparable_chain_with_self()[0].id] for node in nodes}

    return nodes2lft, nodes2lst


def lft_prioritization(wg: WorkGraph, node_id2duration: dict[str, int]) -> list[GraphNode]:
    """
    Return nodes ordered by MIN-LFT priority rule.
    """
    nodes2lft, _ = map_lft_lst(wg, node_id2duration)

    ordered_nodes = sorted(nodes2lft.keys(), key=lambda node_id: nodes2lft[node_id], reverse=True)

    ordered_nodes = [wg[node_id] for node_id in ordered_nodes]

    return ordered_nodes


def lft_randomized_prioritization(wg: WorkGraph, node_id2duration: dict[str, int],
                                  rand: random.Random = random.Random()) -> list[GraphNode]:
    """
    Return ordered nodes sampled by MIN-LFT and MIN-LST priority rules.
    """

    def is_eligible(node_id):
        return parents[node_id].issubset(selected_ids_set)

    nodes2lft, nodes2lst = map_lft_lst(wg, node_id2duration)

    nodes = [wg[node_id] for node_id in nodes2lft.keys()]

    # map nodes and inseparable chain's head
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child.id] = node.id

    # map nodes and their children using only inseparable chain heads
    children = {node.id: set([inseparable_parents[child.id]
                              for inseparable in node.get_inseparable_chain_with_self()
                              for child in inseparable.children]) - {node.id}
                for node in nodes}

    # map nodes and their parents using only inseparable chain heads
    parents = {node.id: set() for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].add(node)

    selected_ids = []
    selected_ids_set = set()
    candidates = {wg.start.id}

    while candidates:
        eligibles = [node_id for node_id in candidates if is_eligible(node_id)]

        priority_mapper = nodes2lft if rand.random() < 0.5 else nodes2lst

        weights = np.array([priority_mapper[node_id] for node_id in eligibles])
        weights = weights.max() - weights + 1
        weights = weights / weights.sum()

        selected_id = rand.choices(eligibles, weights=weights)[0]

        selected_ids.append(selected_id)
        selected_ids_set.add(selected_id)
        candidates.remove(selected_id)
        candidates.update([child_id for child_id in children[selected_id]])

    ordered_nodes = list(reversed([wg[node_id] for node_id in selected_ids]))

    return ordered_nodes
