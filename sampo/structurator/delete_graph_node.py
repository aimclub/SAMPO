from sampo.schemas.graph import WorkGraph
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy, new_start_finish


def delete_graph_node(original_wg: WorkGraph, remove_gn_id: str, change_id: bool = True) -> WorkGraph:
    """
    Deletes a task from Work Graph.
    If the task consists of several inseparable nodes this function deletes all of those nodes
    :param original_wg: Work Graph from which a task is deleted
    :param remove_gn_id: ID of the node, corresponding to the deleted task.
    If the task consists of several inseparable nodes, this is ID of one of them
    :param change_id: Do IDs in the new graph need to be changed
    :return: New WorkGraph with deleted task
    """
    copied_nodes, old_to_new_ids = prepare_work_graph_copy(original_wg, change_id=change_id)

    copied_remove_gn = copied_nodes[old_to_new_ids[remove_gn_id]]

    inseparable_chain = copied_remove_gn.get_inseparable_chain()
    if inseparable_chain is not None:
        copied_remove_gn = inseparable_chain[len(inseparable_chain) - 1]

    parents = copied_remove_gn.parents
    children = copied_remove_gn.children

    parent_to_delete = copied_remove_gn.inseparable_parent

    for parent in parents:
        for edge in parent.edges_from:
            if edge.finish.id == copied_remove_gn.id:
                parent.edges_from.remove(edge)

    for child in children:
        for edge in child.edges_to:
            if edge.start.id == copied_remove_gn.id:
                child.edges_to.remove(edge)

    copied_nodes.pop(copied_remove_gn.id)

    for child in children:
        child.add_parents(parents)

    start, finish = new_start_finish(original_wg, copied_nodes, old_to_new_ids)

    for node in copied_nodes.values():
        node.__dict__.pop('parents', None)
        node.__dict__.pop('parents_set', None)
        node.__dict__.pop('children', None)
        node.__dict__.pop('children_set', None)
        node.__dict__.pop('inseparable_parent', None)
        node.__dict__.pop('inseparable_son', None)
        node.__dict__.pop('get_inseparable_chain', None)

    if parent_to_delete is not None:
        return delete_graph_node(WorkGraph(start, finish), parent_to_delete.id, change_id)

    return WorkGraph(start, finish)
