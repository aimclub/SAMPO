def test_destructor_wg(setup_simple_synthetic):
    wg = setup_simple_synthetic.work_graph(top_border=1)
    print(wg.nodes)
    print(wg.vertex_count)
    del wg
