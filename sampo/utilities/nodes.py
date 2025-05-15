from sampo.schemas import GraphNode, EdgeType


def copy_nodes(nodes: list[GraphNode], drop_outer_works: bool = False) -> list[GraphNode]:
    """
    Returns a list of copied nodes.
    For the right work this list should not have edges with the outer world.
    """
    serialized_nodes = [node._serialize() for node in nodes]
    deserialized_nodes = [GraphNode._deserialize(node_repr) for node_repr in serialized_nodes]

    nodes_dict = dict()
    for node_info in deserialized_nodes:
        wu, parent_info = (node_info[member] for member in ('work_unit', 'parent_edges'))
        if drop_outer_works:
            predecessors = [(nodes_dict[p_id], p_lag, p_type)
                            for p_id, p_lag, p_type in parent_info
                            if p_id in nodes_dict]
        else:
            predecessors = [(nodes_dict[p_id], p_lag, p_type) for p_id, p_lag, p_type in parent_info]
        graph_node = GraphNode(wu, predecessors)
        # TODO Hate
        # if drop_outer_works:
        #     graph_node.children = [node for node in graph_node.children if node.id in nodes_dict]
        nodes_dict[wu.id] = graph_node

    return list(nodes_dict.values())


def add_default_predecessor(nodes: list[GraphNode], predecessor: GraphNode):
    for node in nodes:
        # if not node.parents:
        node.add_parents([predecessor])


def insert_nodes_between(nodes: list[GraphNode], starts: list[GraphNode], finishes: list[GraphNode]):
    without_successors = []
    for node in nodes:
        if len(node.parents) == 0:
            node.add_parents(starts)
        if len(node.children) == 0:
            without_successors.append(node)

    for finish in finishes:
        finish.add_parents(without_successors)
