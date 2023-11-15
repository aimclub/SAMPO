from sampo.structurator.delete_graph_node import delete_graph_node


def test_delete_graph_node(setup_wg):
    remove_node_id = setup_wg.nodes[3].id
    new_wg = delete_graph_node(setup_wg, remove_node_id, change_id=False)

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

    assert not is_node_in_wg, 'Node is still in wg.nodes'
    assert not is_node_someones_parent, 'Node is still someones parent'
    assert not is_node_someones_child, 'Node is still someones child'
