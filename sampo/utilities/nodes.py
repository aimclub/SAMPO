from sampo.schemas import GraphNode


def copy_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    """
    Returns a list of copied nodes.
    For the right work this list should not have edges with the outer world.
    """
    serialized_nodes = [node._serialize() for node in nodes]
    deserialized_nodes = [GraphNode._deserialize(node_repr) for node_repr in serialized_nodes]

    nodes_dict = dict()
    for node_info in deserialized_nodes:
        wu, parent_info = (node_info[member] for member in ('work_unit', 'parent_edges'))
        graph_node = GraphNode(wu, [(nodes_dict[p_id], p_lag, p_type) for p_id, p_lag, p_type in parent_info])
        nodes_dict[wu.id] = graph_node

    return nodes
