from sampo.structurator.delete_graph_node import delete_graph_node


# TODO rewrite test
# TODO docstring documentation

def test_delete_graph_node(setup_wg):
    remove_node_id = setup_wg.nodes[3].id
    new_wg = delete_graph_node(setup_wg, remove_node_id)

    # assert len(new_wg.nodes) == len(setup_wg.nodes) - 1

    is_node_in_wg = False
    is_node_someones_parent = False
    is_node_someones_child = False
    for node in new_wg.nodes:
        if node.id == remove_node_id:
            is_node_in_wg = True
        for parent in node.parents:
            if parent.id == remove_node_id:
                is_node_someones_parent = True
        for child in node.children:
            if child.id == remove_node_id:
                is_node_someones_child = True

    assert not is_node_in_wg
    assert not is_node_someones_parent
    assert not is_node_someones_child
