from sampo.schemas.graph import WorkGraph
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy, new_start_finish


def delete_graph_node(original_wg: WorkGraph, remove_gn_id: str) -> WorkGraph:
    copied_nodes, old_to_new_ids = prepare_work_graph_copy(original_wg)

    copied_remove_gn = copied_nodes[old_to_new_ids[remove_gn_id]]

    parents = copied_remove_gn.parents
    children = copied_remove_gn.children

    for parent in parents:
        for edge in parent.edges_from:
            if edge.finish.id == copied_remove_gn.id:
                parent.edges_from.remove(edge)

    for child in children:
        for edge in child.edges_to:
            if edge.start.id == copied_remove_gn.id:
                child.edges_to.remove(edge)

    for child in children:
        child.add_parents(parents)

    copied_nodes.pop(copied_remove_gn.id)

    start, finish = new_start_finish(original_wg, copied_nodes, old_to_new_ids)

    return WorkGraph(start, finish)
