from operator import itemgetter
from sampo.schemas.graph import WorkGraph
from sampo.generator.pipeline.project import get_start_stage, get_finish_stage


def data_to_work_graph(obj):
    no_parent, no_son = make_extreme_lists(obj)
    no_parent_nodes = []
    no_son_nodes = []
    all_nodes = get_all_nodes(obj)
    node_parents = {}
    node_by_id = {}

    # don't account hierarchical works
    for node, parent_ids in (i.graph_node_to_sampo() for i in all_nodes if not i.is_service):
        node_parents[node.id] = parent_ids
        node_by_id[node.id] = node

        if node.id in no_parent:
            no_parent_nodes.append(node)
        if node.id in no_son:
            no_son_nodes.append(node)

    # fill parents only for those who have them
    for node in filter(lambda n: n not in no_parent_nodes, node_by_id.values()):
        p = itemgetter(*node_parents[node.id])(node_by_id)
        # itemgetter can return single or multiple items
        node.add_parents(list(p) if isinstance(p, tuple) else [p])

    start = get_start_stage()
    for work in no_parent_nodes:
        work.add_parents(parent_works=[start])

    return WorkGraph(start=start, finish=get_finish_stage(no_son_nodes))