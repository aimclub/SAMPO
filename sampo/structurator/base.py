from copy import deepcopy
from typing import Optional
from operator import itemgetter
from math import ceil

from sampo.schemas.graph import GraphNode, GraphNodeDict, GraphEdge, WorkGraph, EdgeType
from sampo.schemas.works import WorkUnit
from sampo.schemas.requirements import WorkerReq, MaterialReq, ConstructionObjectReq, EquipmentReq

STAGE_SEP = '_stage_'


def make_new_node_id(work_unit_id: str, ind: int) -> str:
    """
    Creates an auxiliary id for restructuring the graph

    :param work_unit_id: str - id of work unit
    :param ind: int - sequence number of work_unit stage
    :return:
       auxiliary_id: str - an auxiliary id for the work unit
    """
    return f'{work_unit_id}{STAGE_SEP}{ind}'


def fill_parents_to_new_nodes(origin_node: GraphNode, id2new_nodes: GraphNodeDict,
                              restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
                              use_lag_edge_optimization: bool):
    """
    Restores edges for transformed node

    :param origin_node: GraphNode - The original unconverted node
    :param id2new_nodes: GraphNodeDict - Dictionary with restructured new nodes id
        where the restructured nodes are written
    :param restructuring_edges2new_nodes_id: dict[tuple[str, str], str] - Dictionary for matching edges in the original
        work graph and IDs of new nodes that logically match those edges
    :param use_lag_edge_optimization: bool - if true - considers lags amount in edges,
        otherwise considers lags equal to zero and LagFinishStart edges as FinishStart edges
    :return: Nothing
    """
    last_stage_id = origin_node.id
    zero_stage_id = make_new_node_id(origin_node.id, 0)
    zero_stage_id = zero_stage_id if zero_stage_id in id2new_nodes else last_stage_id

    parents_zero_stage: list[tuple[GraphNode, float, EdgeType]] = []
    parents_last_stage: list[tuple[GraphNode, float, EdgeType]] = []
    for edge in origin_node.edges_to:
        indent = 1 if not (edge.start.work_unit.is_service_unit or edge.finish.work_unit.is_service_unit) else 0
        if edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart]:
            lag = edge.lag if not edge.lag % 1 else ceil(edge.lag)
            lag = lag if lag > 0 else indent
            parents_zero_stage.append((id2new_nodes[edge.start.id], lag, edge.type))
        elif not use_lag_edge_optimization:
            match edge.type:
                case EdgeType.StartStart:
                    new_parent_node_id = make_new_node_id(edge.start.id, 0)
                    parents_zero_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    parents_last_stage.append((id2new_nodes[edge.start.id], indent, EdgeType.FinishStart))
                case EdgeType.LagFinishStart:
                    parents_zero_stage.append((id2new_nodes[edge.start.id], indent, EdgeType.FinishStart))
        else:
            match edge.type:
                case EdgeType.StartStart:
                    new_parent_node_id = restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))
                    parents_zero_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    stage_node = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    stage_node.add_parents([(id2new_nodes[edge.start.id], indent, EdgeType.FinishStart)])
                case EdgeType.LagFinishStart:
                    new_parent_node_id = restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))
                    parents_zero_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                    stage_node = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    stage_node.add_parents([(id2new_nodes[edge.start.id], indent, EdgeType.FinishStart)])
    id2new_nodes[zero_stage_id].add_parents(parents_zero_stage)
    id2new_nodes[last_stage_id].add_parents(parents_last_stage)


def split_node_into_stages(origin_node: GraphNode, restructuring_edges: list[tuple[GraphEdge, bool]],
                           id2new_nodes: GraphNodeDict,
                           restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
                           use_lag_edge_optimization: bool):
    """
    Splits the node into stages according to the lags in the restructuring edges.
    For all stages except the last one, the id changes. For the last one, the id remains the same,
    so that it is more convenient to restore the edges.
    The resulting nodes are chained together by Inseparable-Finish-Start edges.

    :param restructuring_edges: list[tuple[GraphEdge, bool]] - list of restructuring edges and bool flag of reversion
    :param origin_node: GraphNode - Node to be divided into stages
    :param id2new_nodes: GraphNodeDict - Dictionary with restructured new nodes id where the restructured nodes will
        be written
    :param restructuring_edges2new_nodes_id: dict[tuple[str, str], str] - Dictionary for matching edges in the original
        work graph and IDs of new nodes that logically match those edges
    :param use_lag_edge_optimization: bool - if true - considers lags amount in edges,
        otherwise considers lags equal to zero and LagFinishStart edges as FinishStart edges
    :return: Nothing
        """

    def get_reqs_amounts(volume_proportion: float, reqs_amounts_accum: dict[str, list[int]]):
        reqs_amounts = {}
        for reqs in reqs2classes:
            attr = 'volume' if reqs == 'worker_reqs' else 'count'
            new_amounts = [int(volume_proportion * getattr(req, attr)) for req in getattr(wu, reqs)]
            reqs_amounts[reqs] = new_amounts
            reqs_amounts_accum[reqs] = [accum_amount + amount
                                        for accum_amount, amount in zip(reqs_amounts_accum[reqs], new_amounts)]
        return reqs_amounts, reqs_amounts_accum

    def make_new_stage_node(volume_proportion: float,
                            edge_with_pred_stage_node: list[tuple['GraphNode', float, EdgeType]],
                            wu_attrs: dict,
                            reqs2attrs: dict[str, list[dict]]):
        new_reqs = {}
        for reqs, req_class in reqs2classes.items():
            attr = 'volume' if reqs == 'worker_reqs' else 'count'
            for req_attrs, amount in zip(reqs2attrs[reqs], reqs_amounts[reqs]):
                req_attrs[attr] = amount
            new_reqs[reqs] = [req_class(**attrs) for attrs in reqs2attrs[reqs]]
        wu_attrs.update(new_reqs)
        wu_attrs['id'] = stage_node_id
        wu_attrs['name'] = f'{wu.name}{STAGE_SEP}{stage_i}'
        wu_attrs['volume'] = wu.volume * volume_proportion
        new_wu = WorkUnit(**wu_attrs)
        return GraphNode(new_wu, edge_with_pred_stage_node)

    def match_pred_restructuring_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id: dict[tuple[str, str], str]):
        for pred_edge, pred_is_edge_to_node in edges_to_match_with_stage_nodes:
            start, finish = pred_edge.start.id, pred_edge.finish.id
            if pred_is_edge_to_node:
                start, finish = finish, start
            restructuring_edges2new_nodes_id[(start, finish)] = \
                pred_stage_node_id if not pred_is_edge_to_node else stage_node_id

    wu = origin_node.work_unit
    wu_attrs = dict(wu.__dict__)
    if not restructuring_edges:
        id2new_nodes[wu.id] = GraphNode(deepcopy(wu), [])
        return
    reqs2classes = {'worker_reqs': WorkerReq, 'equipment_reqs': EquipmentReq,
                    'object_reqs': ConstructionObjectReq, 'material_reqs': MaterialReq}
    reqs2attrs = {reqs: [dict(req.__dict__) for req in getattr(wu, reqs)] for reqs in reqs2classes}
    reqs_amounts_accum = {reqs: [0 for _ in getattr(wu, reqs)] for reqs in reqs2classes}
    proportions_accum = [(int(is_edge_to_node) +
                          ((1 - 2 * int(is_edge_to_node)) * edge.lag / edge.start.work_unit.volume
                           if use_lag_edge_optimization and 0 < edge.lag <= edge.start.work_unit.volume
                           else 0
                           ),
                          edge, is_edge_to_node
                          )
                         for edge, is_edge_to_node in restructuring_edges]
    proportions_accum.sort(key=itemgetter(0))
    stage_i = 0
    accum, edge, is_edge_to_node = proportions_accum[0]
    stage_node_id = make_new_node_id(wu.id, stage_i)
    reqs_amounts, reqs_amounts_accum = get_reqs_amounts(accum, reqs_amounts_accum)
    id2new_nodes[stage_node_id] = make_new_stage_node(accum, [], wu_attrs, reqs2attrs)
    edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)] if use_lag_edge_optimization else None
    for value, pred_value in zip(proportions_accum[1:], proportions_accum):
        accum, edge, is_edge_to_node = value
        accum_pred, _, _ = pred_value
        if accum == accum_pred:
            if use_lag_edge_optimization:
                edges_to_match_with_stage_nodes.append((edge, is_edge_to_node))
            continue
        stage_i += 1
        pred_stage_node_id = stage_node_id
        stage_node_id = make_new_node_id(wu.id, stage_i)
        proportion = accum - accum_pred
        reqs_amounts, reqs_amounts_accum = get_reqs_amounts(proportion, reqs_amounts_accum)
        id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[pred_stage_node_id], 1,
                                                                        EdgeType.InseparableFinishStart)],
                                                          wu_attrs, reqs2attrs
                                                          )
        if use_lag_edge_optimization:
            match_pred_restructuring_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id)
            edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)]
    stage_i += 1
    pred_stage_node_id = stage_node_id
    stage_node_id = wu.id
    proportion = 1 - accum
    for reqs in reqs2classes:
        attr = 'volume' if reqs == 'worker_reqs' else 'count'
        reqs_amounts[reqs] = [getattr(req, attr) - req_accum
                              for req, req_accum in zip(getattr(wu, reqs), reqs_amounts_accum[reqs])]
    id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[pred_stage_node_id], 1,
                                                                    EdgeType.InseparableFinishStart)],
                                                      wu_attrs, reqs2attrs
                                                      )
    if use_lag_edge_optimization:
        match_pred_restructuring_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id)


def graph_restructuring(wg: WorkGraph, use_lag_edge_optimization: Optional[bool] = False) -> WorkGraph:
    """
    Rebuilds all edges into Finish-Start and Inseparable-Finish-Start edges
    with the corresponding rebuilding of the nodes and returns new work graph

    :param wg: WorkGraph - The graph to be converted
    :param use_lag_edge_optimization: bool - if true - considers lags amount in edges,
        otherwise considers lags equal to zero and LagFinishStart edges as FinishStart edges
    :return:
        new_work_graph: WorkGraph - restructured graph
    """

    def get_restructuring_edges(edges: list[GraphEdge], edge_type: EdgeType, is_edge_to_node: bool):
        return [(edge, is_edge_to_node) for edge in edges if edge.type is edge_type]

    id2new_nodes: GraphNodeDict = dict()
    restructuring_edges2new_nodes_id: dict[tuple[str, str], str] = dict()
    for node in wg.nodes:
        restructuring_edges = get_restructuring_edges(node.edges_from, EdgeType.StartStart, False) + \
                              get_restructuring_edges(node.edges_to, EdgeType.FinishFinish, True)
        if use_lag_edge_optimization:
            restructuring_edges += get_restructuring_edges(node.edges_to, EdgeType.LagFinishStart, True) + \
                                   get_restructuring_edges(node.edges_from, EdgeType.LagFinishStart, False)
        split_node_into_stages(node, restructuring_edges, id2new_nodes, restructuring_edges2new_nodes_id,
                               use_lag_edge_optimization)
        fill_parents_to_new_nodes(node, id2new_nodes, restructuring_edges2new_nodes_id, use_lag_edge_optimization)
    return WorkGraph(id2new_nodes[wg.start.id], id2new_nodes[wg.finish.id])
