from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Set

from sampo.schemas.graph import GraphNode, GraphNodeDict, GraphEdge, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.works import WorkUnit


def make_start_id(work_unit_id: str, ind: int) -> str:
    """
    Creates an auxiliary id for restructuring the graph
    :param work_unit_id: str - work unit id
    :param ind: int - sequence number of work_unit stage
    :return:
       auxiliary_id: str - an auxiliary id for the work unit
    """
    return work_unit_id + f"_stage_{ind}"


def find_lags(edges: List[GraphEdge], edge_type: EdgeType, is_reversed: bool) -> (List[Tuple[float, float, bool]]):
    """
    Searches for the maximum lag among the given edges and type of edge,
        for the maximum lag saves the amount of parental work
    :param is_reversed: bool - Used to specify from which node lag is used
    :param edges: List[GraphEdge] - the given edges
    :param edge_type: str - the given type of edge
    :return:
        max_lag: float - the maximum lag
        parent_volume: float - The volume for which the lag is set, i.e. the parent volume
    """
    lag_volume_list = [(edge.lag, edge.start.work_unit.volume, is_reversed)
                       for edge in edges if edge_type is edge.type]
    return lag_volume_list


def node_restructuring(origin_node: GraphNode, id2new_nodes: GraphNodeDict,
                       lags_volumes_list: List[Tuple[float, float, bool]],
                       old_id_lag2new_id: Dict[Tuple[str, float, bool], str]):
    """
    Splits the node into two parts: into "piece" and "mian" whose sizes are proportional to max_lag and
    parent_volume-max_lag, respectively. The order of "piece" and "mian" is set. For the first vertex the id is changed,
    for the second one it remains the same, so that it is more convenient to restore the edges. The resulting nodes are
    created without edges. It connects two obtained nodes with an unbreakable edge,
    which does not allow to perform tasks in any way
    :param lags_volumes_list: List[Tuple[float, float, bool]] -
    :param origin_node: GraphNode - Node to be divided into two parts
    :param id2new_nodes: GraphNodeDict - Dictionary with restructured new nodes where the restructured nodes will
        be written
    :param old_id_lag2new_id: Dict[Tuple[str, float], str]
    :return: Nothing
    """
    wu = origin_node.work_unit
    if len(lags_volumes_list) == 0 or wu.is_service_unit:
        id2new_nodes[wu.id] = GraphNode(deepcopy(wu), [])
        return

    # last elem - whole work_unit
    proportions_accum = sorted(
        [(1 * int(is_reversed) + lag / volume * (1 - 2 * int(is_reversed)), lag, is_reversed)
         for lag, volume, is_reversed in lags_volumes_list
         if volume > 0 and lag < volume] + [(1, -1, False)])

    proportions_accum.sort()
    proportions: List[Tuple[float, float, bool]] = [proportions_accum[0]]
    for ind in range(1, len(proportions_accum)):
        accum, lag, is_reversed = proportions_accum[ind]
        accum_pred, _, _ = proportions_accum[ind - 1]
        if accum == accum_pred and (ind != len(proportions_accum) - 1):
            continue
        proportions.append((accum - accum_pred, lag, is_reversed))

    for ind in range(len(proportions)):
        piece_div_main, lag, is_reversed = proportions[ind]
        reqs = [WorkerReq(wr.kind, wr.volume * piece_div_main, wr.min_count, wr.max_count)
                for wr in wu.worker_reqs]

        volume = wu.volume * piece_div_main

        new_id = make_start_id(wu.id, ind) if ind < len(proportions) - 1 else wu.id
        new_wu = WorkUnit(new_id, wu.name + f"_stage_{ind}", reqs, group=wu.group,
                          volume=volume, volume_type=wu.volume_type)
        parents = [(id2new_nodes[make_start_id(wu.id, ind - 1)], 0, EdgeType.InseparableFinishStart)] if ind > 0 else []
        id2new_nodes[new_id] = GraphNode(new_wu, parents)
        old_id_lag2new_id[(wu.id, lag, is_reversed)] = new_id
    return


def fill_parents(origin_work_graph: WorkGraph, id2new_nodes: GraphNodeDict,
                 old_id_lag2new_id: Dict[Tuple[str, float, bool], str],
                 use_ffs_separately: bool = False):
    """
    Restores edges in the transformed graph

    :param origin_work_graph: WorkGraph - The original unconverted graph
    :param id2new_nodes: GraphNodeDict -dictionary with transformed vertices,
        if some vertices from the original graph are missing,
        the function converts their copy and writes it into the dictionary
    :param old_id_lag2new_id: Dict[Tuple[str, float, bool], str]
    :param use_ffs_separately:
        If false, then FFS edges are considered equivalent to FS,
        otherwise they are converted as a separate type of edges
    :return: Nothing
    """

    # from the first element since the zero node is the starting node that has no parents
    for node in origin_work_graph.nodes[1:]:
        zero_stage_id = make_start_id(node.id, 0) if make_start_id(node.id, 0) in id2new_nodes else node.id
        last_stage_id = node.id

        parents_zero_stage: List[Tuple[GraphNode, float, EdgeType]] = []
        parents_last_stage: List[Tuple[GraphNode, float, EdgeType]] = []
        for edge in node.edges_to:
            if edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart] or \
                    not use_ffs_separately and edge.type is EdgeType.LagFinishStart:
                parents_zero_stage.append((id2new_nodes[edge.start.id], edge.lag, edge.type))
            elif edge.type is EdgeType.StartStart:
                work_id = make_start_id(edge.start.id, 0)
                work_id = work_id if work_id in id2new_nodes else edge.start.id
                parents_zero_stage.append((id2new_nodes[work_id], edge.lag, EdgeType.FinishStart))
            elif edge.type is EdgeType.FinishFinish:
                parents_last_stage.append((id2new_nodes[edge.start.id], edge.lag, EdgeType.FinishStart))
            elif use_ffs_separately and edge.type is EdgeType.LagFinishStart:
                parents_zero_stage.append(
                    (id2new_nodes[old_id_lag2new_id[edge.start.id, edge.lag, False]], 0, EdgeType.FinishStart))
                id2new_nodes[last_stage_id].add_parents([(id2new_nodes[edge.start.id], 0, EdgeType.FinishStart)])
        id2new_nodes[zero_stage_id].add_parents(parents_zero_stage)
        id2new_nodes[last_stage_id].add_parents(parents_last_stage)

        # add SF connections from origin graph
        parents = [(id2new_nodes[edge.finish.id], edge.lag, EdgeType.FinishStart)
                   for edge in node.edges_from if edge.type is EdgeType.StartFinish]
        # after reversing the start-finish node could remain without a parent, if so, suspend it in the start node
        # parents = parents or [id2new_nodes[origin_work_graph.start.id]]
        id2new_nodes[zero_stage_id].add_parents(parents)

    start_id = origin_work_graph.start.id
    finish_id = origin_work_graph.finish.id
    has_child: Set[str] = {start_id, finish_id}
    has_parent: Set[str] = {start_id, finish_id}
    for node in id2new_nodes.values():
        for edge in node.edges_to:
            has_child.add(edge.start.id)
            has_parent.add(edge.finish.id)

    has_no_child = set(id2new_nodes.keys()) - has_child
    id2new_nodes[origin_work_graph.finish.id].add_parents([id2new_nodes[node_id] for node_id in has_no_child])

    start_id = origin_work_graph.start.id
    has_no_parent = set(id2new_nodes.keys()) - has_parent
    for work_id in has_no_parent:
        id2new_nodes[work_id].add_parents([id2new_nodes[start_id]])


def graph_restructuring(wg: WorkGraph, use_lag_edge_optimization: Optional[bool] = False) -> WorkGraph:
    """
    Rebuilds all edges into finish-start edges with the corresponding rebuilding of the nodes

    :param wg: WorkGraph - The graph to be converted
    :param use_lag_edge_optimization: bool - if true - do optimization fake-finish-start edges,
        otherwise considers such edges to be similar to finish-start
    :return:
        new_work_graph: WorkGraph - restructured graph
    """
    id2new_nodes: GraphNodeDict = dict()
    old_id_lag2new_od: Dict[Tuple[str, float, bool], str] = dict()
    for node in wg.nodes:
        lags_volumes = []
        lags_volumes += find_lags(node.edges_from, EdgeType.StartStart, False)
        lags_volumes += find_lags(node.edges_to, EdgeType.FinishFinish, True)
        if use_lag_edge_optimization:
            lags_volumes += find_lags(node.edges_to, EdgeType.LagFinishStart, True)
            lags_volumes += find_lags(node.edges_from, EdgeType.LagFinishStart, False)
        node_restructuring(node, id2new_nodes, lags_volumes, old_id_lag2new_od)
    fill_parents(wg, id2new_nodes, old_id_lag2new_od, use_ffs_separately=use_lag_edge_optimization)
    return WorkGraph(id2new_nodes[wg.start.id], id2new_nodes[wg.finish.id])
