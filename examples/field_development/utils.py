from operator import itemgetter
from sampo.schemas.graph import WorkGraph
from sampo.generator.pipeline.project import get_start_stage, get_finish_stage

import logging

logger = logging.getLogger("uvicorn")


def get_all_nodes(graph):
    list_of_nodes = []
    for i in graph['upper_works']:
        if not i['is_service']:
            list_of_nodes.append(i)
        list_of_nodes.extend(get_nodes(i))
    return list_of_nodes


def get_nodes(item):
    local_list = []
    for i in item.children:
        if not i.is_service:
            local_list.append(i)
        local_list.extend(get_nodes(i))
    return local_list


def get_all_indexes(graph):
    list_of_ind = []
    for i in graph.upper_works:
        list_of_ind.append(i.outer_id)
        list_of_ind.extend(get_index(i))
    return list_of_ind


def get_index(item):
    local_list = []
    for i in item.children:
        local_list.append(i.outer_id)
        local_list.extend(get_index(i))
    return local_list


def edges_indexes(list_of_nodes):
    parent = []
    son = []
    for i in list_of_nodes:
        for j in i.edges:
            parent.append(j.parent_id)  # все вершины, которые являются родителями
            son.append(j.son_id)  # все вершины, которые являются наследниками

    parent = list(set(parent))
    son = list(set(son))
    return parent, son


def make_extreme_lists(graph):
    all_nodes = get_all_nodes(graph)
    all_index = get_all_indexes(graph)
    parent, son = edges_indexes(all_nodes)
    no_son = list_differences(all_index, parent)  # все вершины без наследников
    no_parent = list_differences(all_index, son)  # все вершины без детей
    return no_parent, no_son


def list_differences(hole_list, part_list):
    exit = []
    for i in hole_list:
        if not i in part_list:
            exit.append(i)
    return exit


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