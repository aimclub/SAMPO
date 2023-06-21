from sampo.schemas.graph import GraphNode, EdgeType, WorkGraph, GraphEdge
from sampo.schemas.works import WorkUnit
from sampo.structurator.graph_insertion import prepare_work_graph_copy


def insert_work_unit(original_wg: WorkGraph, inserted_wu: WorkUnit,
                     parents_edges: list[tuple[GraphNode, EdgeType]],
                     children_edges: list[tuple[GraphNode, EdgeType]]) -> WorkGraph:
    original_nodes, original_old_to_new_ids = prepare_work_graph_copy(original_wg, [])

    new_parents_edges = [(original_nodes[original_old_to_new_ids[parent.id]],  edge_type)
                         for parent, edge_type in parents_edges]
    new_children_edges = [(original_nodes[original_old_to_new_ids[child.id]], edge_type)
                          for child, edge_type in children_edges]

    new_node = GraphNode(inserted_wu, [])
    new_node.add_parents(new_parents_edges)
    for child, edge in new_children_edges:
        child.add_parents(([new_node], edge))

    new_start = original_nodes[original_old_to_new_ids[original_wg.start.id]]
    new_finish = original_nodes[original_old_to_new_ids[original_wg.finish.id]]

    return WorkGraph(new_start, new_finish)

