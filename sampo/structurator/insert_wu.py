from sampo.schemas.graph import GraphNode, EdgeType, WorkGraph
from sampo.schemas.works import WorkUnit
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy, new_start_finish


def insert_work_unit(original_wg: WorkGraph, inserted_wu: WorkUnit,
                     parents_edges: list[GraphNode] or list[tuple[GraphNode, float, EdgeType]],
                     children_edges: list[GraphNode] or list[tuple[GraphNode, float, EdgeType]],
                     change_id: bool = True) -> WorkGraph:
    """
    Inserts new node, based on given Work Unit, in the Work Graph
    :param original_wg: WorkGraph into which we insert new node
    :param inserted_wu: WorkUnit on the basis of which we create new GraphNode
    :param parents_edges: Nodes which are supposed to be the parents of new GraphNode
    :param children_edges: Nodes which are supposed to be the children of new GraphNode
    :param change_id: Do IDs in the new graph need to be changed
    :return: New WorkGraph with inserted new node
    """
    copied_nodes, original_old_to_new_ids = prepare_work_graph_copy(original_wg, change_id=change_id)

    new_parents_edges = _new_edges(copied_nodes, original_old_to_new_ids, parents_edges)
    new_children_edges = _new_edges(copied_nodes, original_old_to_new_ids, children_edges)

    new_node = GraphNode(inserted_wu, [])
    new_node.add_parents(new_parents_edges)
    if isinstance(new_children_edges[0], GraphNode):
        for child in new_children_edges:
            child.add_parents([new_node])
    elif isinstance(children_edges[0], tuple):
        for child, lag, edge in new_children_edges:
            child.add_parents([(new_node, lag, edge)])

    new_start, new_finish = new_start_finish(original_wg, copied_nodes, original_old_to_new_ids)

    return WorkGraph(new_start, new_finish)


def _new_edges(copied_nodes: dict[str, GraphNode], original_old_to_new_ids: dict[str, str],
               edges: list[GraphNode] or list[tuple[GraphNode, float, EdgeType]]) \
        -> list[GraphNode] or list[tuple[GraphNode, float, EdgeType]]:
    if isinstance(edges[0], GraphNode):
        return [copied_nodes[original_old_to_new_ids[parent.id]] for parent in edges]
    elif isinstance(edges[0], tuple):
        return [(copied_nodes[original_old_to_new_ids[parent.id]], lag, edge_type)
                for parent, lag, edge_type in edges]
    else:
        return []
