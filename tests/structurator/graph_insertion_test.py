from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy


def test_prepare_wg_copy(setup_wg):
    copied_nodes, old_to_new_ids = prepare_work_graph_copy(setup_wg)

    assert len(copied_nodes) == len(setup_wg.nodes)

    is_copied_wg_equals_setup_wg = True
    for node in setup_wg.nodes:
        if not (node.work_unit.name == copied_nodes[old_to_new_ids[node.id]].work_unit.name):
            is_copied_wg_equals_setup_wg = False
    assert is_copied_wg_equals_setup_wg
