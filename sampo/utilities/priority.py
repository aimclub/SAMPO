from sampo.base import SAMPO
from sampo.schemas import WorkGraph, GraphNode


def update_priority(node, priority_value, visited=None):
    if visited is None:
        visited = set()

    visited.add(node)

    node.work_unit.priority = min(node.work_unit.priority, priority_value)

    for parent in getattr(node, "parents", []):
        update_priority(parent, priority_value, visited)


def check_and_correct_priorities(wg: WorkGraph, verbose: bool = True):
    # check priorities
    for node in wg.nodes:
        # if node.is_inseparable_son():
        #     continue
        for parent_node in node.parents:
            if node.work_unit.priority < parent_node.work_unit.priority:
                if verbose:
                    SAMPO.logger.warn(f'Priority of node {node.work_unit.display_name} '
                                      f'is less than node {parent_node.work_unit.name} '
                                      f'conflicting with edges, applying auto-correction')
                update_priority(parent_node, node.work_unit.priority)


def extract_priority_groups_from_nodes(nodes: list[GraphNode]) -> dict[int, list[GraphNode]]:
    groups = {priority: [] for priority in set(node.work_unit.priority for node in nodes)}
    for node in nodes:
        groups[node.work_unit.priority].append(node)
    return groups


def extract_priority_groups_from_indices(priorities: list[int]) -> dict[int, list[int]]:
    groups = {priority: [] for priority in set(priorities)}
    for node, priority in enumerate(priorities):
        groups[priority].append(node)
    return groups
