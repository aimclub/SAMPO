from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy, new_start_finish


def delete_graph_node(original_wg: WorkGraph, remove_gn_id: str, change_id: bool = True) -> WorkGraph:
    """
    Deletes a task from WorkGraph.
    If the task consists of several inseparable nodes this function deletes all of those nodes
    :param original_wg: WorkGraph from which a task is deleted
    :param remove_gn_id: id of the node, corresponding to the deleted task.
    If the task consists of several inseparable nodes, this is id of one of them
    :param change_id: do ids in the new graph need to be changed
    :return: new WorkGraph with deleted task
    """
    copied_nodes, old_to_new_ids = prepare_work_graph_copy(original_wg, change_id=change_id)

    copied_remove_gn = copied_nodes[old_to_new_ids[remove_gn_id]]

    inseparable_chain = copied_remove_gn.get_inseparable_chain()
    if inseparable_chain is not None:
        copied_remove_gn = inseparable_chain[len(inseparable_chain) - 1]
        parent_to_delete = copied_remove_gn.inseparable_parent
        while parent_to_delete is not None:
            copied_remove_gn = parent_to_delete
            parent_to_delete = copied_remove_gn.inseparable_parent
            _node_deletion(copied_remove_gn, copied_nodes)
    else:
        _node_deletion(copied_remove_gn, copied_nodes)

    start, finish = new_start_finish(original_wg, copied_nodes, old_to_new_ids)

    return WorkGraph(start, finish)


def _node_deletion(remove_gn: GraphNode, nodes: dict[str, GraphNode]):
    parents = remove_gn.parents
    children = remove_gn.children

    for parent in parents:
        for edge in parent.edges_from:
            if edge.finish.id == remove_gn.id:
                parent.edges_from.remove(edge)

    for child in children:
        for edge in child.edges_to:
            if edge.start.id == remove_gn.id:
                child.edges_to.remove(edge)

    nodes.pop(remove_gn.id)

    for child in children:
        child.add_parents(parents)
