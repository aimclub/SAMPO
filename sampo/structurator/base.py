from copy import deepcopy
from typing import Optional

from sampo.schemas.graph import GraphNode, GraphNodeDict, GraphEdge, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.works import WorkUnit

STAGE_SEP = '_stage_'


def make_new_node_id(work_unit_id: str, ind: int) -> str:
    """
    Creates an auxiliary id for restructuring the graph

    :param work_unit_id: str - work unit id
    :param ind: int - sequence number of work_unit stage
    :return:
       auxiliary_id: str - an auxiliary id for the work unit
    """
    return f"{work_unit_id}{STAGE_SEP}{ind}"


def fill_parents_to_new_nodes(origin_node: GraphNode, id2new_nodes: GraphNodeDict,
                              restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
                              use_lag_edge_optimization: bool):
    """
    Restores edges for transformed node

    :param origin_node: GraphNode - The original unconverted node
    :param id2new_nodes: GraphNodeDict - dictionary with transformed vertices
    :param restructuring_edges2new_nodes_id: dict[tuple[str, str], str]
    :param use_lag_edge_optimization: bool -
        If false, then FFS edges are considered equivalent to FS,
        otherwise they are converted as a separate type of edges
    :return: Nothing
    """
    # from the first element since the zero node is the starting node that has no parents
    last_stage_id = origin_node.id
    zero_stage_id = make_new_node_id(origin_node.id, 0)
    zero_stage_id = zero_stage_id if zero_stage_id in id2new_nodes else last_stage_id

    parents_zero_stage: list[tuple[GraphNode, float, EdgeType]] = []
    parents_last_stage: list[tuple[GraphNode, float, EdgeType]] = []
    for edge in origin_node.edges_to:
        if edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart]:
            parents_zero_stage.append((id2new_nodes[edge.start.id], 0, edge.type))
        elif not use_lag_edge_optimization:
            match edge.type:
                case EdgeType.StartStart:
                    work_id = make_new_node_id(edge.start.id, 0)
                    parents_zero_stage.append((id2new_nodes[work_id], 0, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    parents_last_stage.append((id2new_nodes[edge.start.id], 0, EdgeType.FinishStart))
                case EdgeType.LagFinishStart:
                    parents_zero_stage.append((id2new_nodes[edge.start.id], 0, EdgeType.FinishStart))
        else:
            match edge.type:
                case EdgeType.StartStart:
                    work_id = restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))
                    parents_zero_stage.append((id2new_nodes[work_id], 0, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    work = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    work.add_parents([(id2new_nodes[edge.start.id], 0, EdgeType.FinishStart)])
                case EdgeType.LagFinishStart:
                    work = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))]
                    parents_zero_stage.append((work, 0, EdgeType.FinishStart))
                    work = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    work.add_parents([(id2new_nodes[edge.start.id], 0, EdgeType.FinishStart)])
    id2new_nodes[zero_stage_id].add_parents(parents_zero_stage)
    id2new_nodes[last_stage_id].add_parents(parents_last_stage)


def cut_node_to_stages(origin_node: GraphNode, restructuring_edges: list[tuple[GraphEdge, bool]],
                       id2new_nodes: GraphNodeDict,
                       restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
                       use_lag_edge_optimization: bool):
    """
    Splits the node into two parts: into "piece" and "mian" whose sizes are proportional to max_lag and
    parent_volume-max_lag, respectively. The order of "piece" and "mian" is set. For the first vertex the id is changed,
    for the second one it remains the same, so that it is more convenient to restore the edges. The resulting nodes are
    created without edges. It connects two obtained nodes with an unbreakable edge,
    which does not allow to perform tasks in any way

    :param restructuring_edges: list[tuple[GraphEdge, bool]] - list of representations of edges by tuples of
        lag, volume and bool flag of reversion
    :param origin_node: GraphNode - Node to be divided into two parts
    :param id2new_nodes: GraphNodeDict - Dictionary with restructured new nodes where the restructured nodes will
        be written
    :param restructuring_edges2new_nodes_id: dict[tuple[str, str], str]
    :param use_lag_edge_optimization: bool - if true - considers lags amount in edges,
        otherwise considers lags equal to zero and LagFinishStart edges as FinishStart
    :return: Nothing
        """

    def make_new_stage_node(volume_proportion, edge_with_pred_stage_node):
        reqs = [WorkerReq(wr.kind, wr.volume * volume_proportion, wr.min_count, wr.max_count) for wr in wu.worker_reqs]
        new_wu = WorkUnit(stage_node_id, f'{wu.name}{STAGE_SEP}{stage_i}', reqs, group=wu.group,
                          volume=wu.volume * volume_proportion, volume_type=wu.volume_type,
                          display_name=wu.display_name,
                          workground_size=wu.workground_size)
        return GraphNode(new_wu, edge_with_pred_stage_node)

    def match_pred_restructuring_edges_with_stage_nodes_id():
        for pred_edge, pred_is_edge_to_node in edges_to_match_with_stage_nodes:
            start, finish = pred_edge.start.id, pred_edge.finish.id
            if pred_is_edge_to_node:
                start, finish = finish, start
            restructuring_edges2new_nodes_id[(start, finish)] = \
                pred_stage_node_id if not pred_is_edge_to_node else stage_node_id

    wu = origin_node.work_unit
    if len(restructuring_edges) == 0 or wu.is_service_unit:
        id2new_nodes[wu.id] = GraphNode(deepcopy(wu), [])
        return
    proportions_accum = sorted(
        [(int(is_edge_to_node) +
          ((1 - 2 * int(is_edge_to_node)) * edge.lag / edge.start.work_unit.volume if use_lag_edge_optimization else 0),
          edge, is_edge_to_node)
         for edge, is_edge_to_node in restructuring_edges
         ]
    )  # if volume > 0 and lag < volume
    stage_i = 0
    accum, edge, is_edge_to_node = proportions_accum[0]
    stage_node_id = make_new_node_id(wu.id, stage_i)
    id2new_nodes[stage_node_id] = make_new_stage_node(accum, [])
    edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)] if use_lag_edge_optimization else None
    for i, value in enumerate(proportions_accum[1:]):
        accum, edge, is_edge_to_node = value
        accum_pred, _, _ = proportions_accum[i]
        if accum == accum_pred:
            if use_lag_edge_optimization:
                edges_to_match_with_stage_nodes.append((edge, is_edge_to_node))
            continue
        stage_i += 1
        pred_stage_node_id = stage_node_id
        stage_node_id = make_new_node_id(wu.id, stage_i)
        proportion = accum - accum_pred
        id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[pred_stage_node_id], 0,
                                                                        EdgeType.InseparableFinishStart)]
                                                          )
        if use_lag_edge_optimization:
            match_pred_restructuring_edges_with_stage_nodes_id()
            edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)]
    stage_i += 1
    pred_stage_node_id = stage_node_id
    stage_node_id = wu.id
    proportion = 1 - accum
    id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[pred_stage_node_id], 0,
                                                                    EdgeType.InseparableFinishStart)]
                                                      )
    if use_lag_edge_optimization:
        match_pred_restructuring_edges_with_stage_nodes_id()


def graph_restructuring(wg: WorkGraph, use_lag_edge_optimization: Optional[bool] = False) -> WorkGraph:
    """
    Rebuilds all edges into finish-start edges with the corresponding rebuilding of the nodes

    :param wg: WorkGraph - The graph to be converted
    :param use_lag_edge_optimization: bool - if true - considers lags amount in edges,
        otherwise considers lags equal to zero and LagFinishStart edges as FinishStart
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
        cut_node_to_stages(node, restructuring_edges, id2new_nodes, restructuring_edges2new_nodes_id,
                           use_lag_edge_optimization)
        fill_parents_to_new_nodes(node, id2new_nodes, restructuring_edges2new_nodes_id, use_lag_edge_optimization)
    return WorkGraph(id2new_nodes[wg.start.id], id2new_nodes[wg.finish.id])
