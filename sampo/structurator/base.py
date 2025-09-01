"""Utilities for restructuring work graphs into stage-based forms.

Инструменты для преобразования рабочих графов в последовательность стадий.
"""

from copy import deepcopy
from typing import Optional
from operator import itemgetter
from math import ceil

from sampo.api.const import STAGE_SEP
from sampo.schemas.graph import GraphNode, GraphEdge, WorkGraph, EdgeType
from sampo.schemas.works import WorkUnit
from sampo.schemas.requirements import WorkerReq, MaterialReq, ConstructionObjectReq, EquipmentReq
from sampo.utilities.priority import check_and_correct_priorities


def make_new_node_id(work_unit_id: str, ind: int) -> str:
    """Creates an auxiliary ID for restructuring the graph.

    Создаёт вспомогательный идентификатор для реструктуризации графа.

    Args:
        work_unit_id (str): ID of the work unit.
            Идентификатор работы.
        ind (int): Sequence number of the work unit stage.
            Порядковый номер этапа работы.

    Returns:
        str: Auxiliary ID for the work unit.
            Вспомогательный идентификатор работы.
    """

    return f"{work_unit_id}{STAGE_SEP}{ind}"


def fill_parents_to_new_nodes(
    origin_node: GraphNode,
    id2new_nodes: dict[str, GraphNode],
    restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
    use_lag_edge_optimization: bool,
) -> None:
    """Restores parent edges for a node split into stages.

    Восстанавливает связи с родительскими узлами для узла, разделённого на стадии.

    Args:
        origin_node (GraphNode): The original unconverted node.
            Исходный узел до преобразования.
        id2new_nodes (dict[str, GraphNode]): Mapping of new node IDs to nodes.
            Сопоставление новых идентификаторов узлам.
        restructuring_edges2new_nodes_id (dict[tuple[str, str], str]):
            Mapping between original edges and IDs of new nodes that replace them.
            Сопоставление исходных рёбер и идентификаторов новых узлов, заменяющих их.
        use_lag_edge_optimization (bool): Whether to account for lags in edges.
            Учитывать ли задержки в рёбрах.

    Returns:
        None: This function modifies ``id2new_nodes`` in-place.
            None: функция изменяет ``id2new_nodes`` на месте.
    """

    # last stage id is equal to original node id
    last_stage_id = origin_node.id
    # make first stage id
    first_stage_id = make_new_node_id(origin_node.id, 0)
    # if first stage id not in id2new_nodes then there are no stages, so then first stage id is equal to last stage id
    first_stage_id = first_stage_id if first_stage_id in id2new_nodes else last_stage_id

    # basic time lag of FinishStart or InseparableFinishStart edges
    indent = 0

    # list of edges to parent nodes of first stage
    parents_first_stage: list[tuple[GraphNode, float, EdgeType]] = []
    # list of edges to parent nodes of last stage
    parents_last_stage: list[tuple[GraphNode, float, EdgeType]] = []
    # iterate over edges to parent nodes of original node
    for edge in origin_node.edges_to:
        # TODO Check indent application
        if edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart]:
            if edge.type is EdgeType.InseparableFinishStart:
                # if edge type is InseparableFinishStart then lag is equal to basic time lag
                lag = indent
            else:
                # if edge type is FinishStart then use time lag of original edge
                lag = edge.lag if not edge.lag % 1 else ceil(edge.lag)
                # lag = lag if lag > 0 else indent
            # edges of types FinishStart or InseparableFinishStart must enter the first stage node
            parents_first_stage.append((id2new_nodes[edge.start.id], lag, edge.type))
        elif not use_lag_edge_optimization:
            # if the use_lag_edge_optimization is False, then adding edges is trivial,
            # since a node can be divided into a maximum of three stages,
            # with the intermediate one not being used in creating edges
            match edge.type:
                case EdgeType.StartStart:
                    # make id of created first stage node of original parent node
                    new_parent_node_id = make_new_node_id(edge.start.id, 0)
                    # add edge between parent's first stage node and first stage of current node
                    parents_first_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    # add edge between parent's created last stage node and last stage of current node
                    parents_last_stage.append((id2new_nodes[edge.start.id], indent, EdgeType.FinishStart))
                case EdgeType.LagFinishStart:
                    # if the use_lag_edge_optimization is False,
                    # then edge of type LagFinishStart is just replaced by an edge of type FinishStart
                    # so, add edge between parent's created last stage node and first stage of current node
                    parents_first_stage.append((id2new_nodes[edge.start.id], indent, EdgeType.FinishStart))
        else:
            match edge.type:
                case EdgeType.StartStart:
                    # get id of corresponding created stage node of original parent node
                    new_parent_node_id = restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))
                    # get parent's stage node by received id and add edge between it and first stage of current node
                    parents_first_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                case EdgeType.FinishFinish:
                    # get id of corresponding stage of current node and get the stage node itself
                    stage_node = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    # add edge between parent's created last stage node and received stage node
                    stage_node.add_parents([(id2new_nodes[edge.start.id], indent, EdgeType.FinishStart)])
                case EdgeType.LagFinishStart:
                    # get id of corresponding created stage node of original parent node
                    new_parent_node_id = restructuring_edges2new_nodes_id.pop((edge.start.id, edge.finish.id))
                    # get parent's stage node by received id and add edge between it and first stage of current node
                    parents_first_stage.append((id2new_nodes[new_parent_node_id], indent, EdgeType.FinishStart))
                    # get id of corresponding stage of current node and get the stage node itself
                    stage_node = id2new_nodes[restructuring_edges2new_nodes_id.pop((edge.finish.id, edge.start.id))]
                    # add edge between parent's created last stage node and received stage node
                    stage_node.add_parents([(id2new_nodes[edge.start.id], indent, EdgeType.FinishStart)])

    # add parents to first stage of current node
    id2new_nodes[first_stage_id].add_parents(parents_first_stage)
    # add parents to last stage of current node
    id2new_nodes[last_stage_id].add_parents(parents_last_stage)


def split_node_into_stages(origin_node: GraphNode, restructuring_edges: list[tuple[GraphEdge, bool]],
                           id2new_nodes: dict[str, GraphNode],
                           restructuring_edges2new_nodes_id: dict[tuple[str, str], str],
                           use_lag_edge_optimization: bool):
    """Splits a work node into sequential stages.

    Разделяет узел работы на последовательные стадии.

    The function creates intermediate nodes according to restructuring edges and
    connects them with ``InseparableFinishStart`` edges. The last stage keeps the
    original node ID to simplify parent restoration.

    Функция создаёт промежуточные узлы в соответствии с рёбрами реструктуризации
    и соединяет их рёбрами ``InseparableFinishStart``. Последняя стадия сохраняет
    исходный идентификатор узла для упрощения восстановления родителей.

    Args:
        origin_node (GraphNode): Node to be divided into stages.
            Узел, который требуется разделить на стадии.
        restructuring_edges (list[tuple[GraphEdge, bool]]):
            Restructuring edges with a flag showing direction.
            Рёбра реструктуризации и флаг направления.
        id2new_nodes (dict[str, GraphNode]): Mapping for storing created nodes.
            Отображение для хранения созданных узлов.
        restructuring_edges2new_nodes_id (dict[tuple[str, str], str]):
            Mapping from original edges to new node IDs.
            Сопоставление исходных рёбер с идентификаторами новых узлов.
        use_lag_edge_optimization (bool): Whether to handle lag edges explicitly.
            Учитывать ли задержки в рёбрах явно.

    Returns:
        None: Function modifies mappings in-place.
            None: функция изменяет отображения на месте.
    """

    def get_reqs_amounts(volume_proportion: float, reqs2amounts_accum: dict[str, list[int]]) \
            -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """
        Creates for the new stage node a mapper
        of work unit requirement names and amounts calculated based on volume_proportion.
        Updates the accumulated amounts in reqs2amounts_accum by adding calculated for the new stage node amounts
        """
        # make a mapper of requirement names and amounts for the new stage node
        reqs2amounts = {}
        for reqs in reqs2classes:
            # define a name of the amount attribute of the requirement
            attr = 'volume' if reqs == 'worker_reqs' else 'count'
            # calculate requirement amounts for the new stage node based on volume_proportion
            new_amounts = [int(volume_proportion * getattr(req, attr)) for req in getattr(wu, reqs)]
            reqs2amounts[reqs] = new_amounts
            # update the accumulated amounts of the requirement
            reqs2amounts_accum[reqs] = [accum_amount + amount
                                        for accum_amount, amount in zip(reqs2amounts_accum[reqs], new_amounts)]
        return reqs2amounts, reqs2amounts_accum

    def make_new_stage_node(volume_proportion: float,
                            edge_with_prev_stage_node: list[tuple[GraphNode, float, EdgeType]],
                            wu_attrs: dict,
                            reqs2attrs: dict[str, list[dict]]) -> GraphNode:
        """
        This function updates the requirements attributes of the original node with new pre-calculated amounts.
        Using these updated attributes, new requirement instances are created
        and the work unit attributes are updated accordingly. Also, in the attributes of the work unit, the id, name
        and volume are updated. Volume calculates based on volume_proportion.
        The resulting attributes are then used to create new Work Unit to make returned new stage Node.

        :param volume_proportion: proportion of volume for new stage
        :param edge_with_prev_stage_node: list with edge to previous stage node
        :param wu_attrs: copied attributes of work unit of original node
        :param reqs2attrs: copied attributes of requirements of original node

        :return: created Graph Node of new stage
        """
        # new requirements to update work unit attributes
        new_reqs = {}
        for reqs, req_class in reqs2classes.items():
            # define a name of the amount attribute of the requirement
            attr = 'volume' if reqs == 'worker_reqs' else 'count'
            for req_attrs, amount in zip(reqs2attrs[reqs], reqs2amounts[reqs]):
                # update amount attribute in requirement attributes
                req_attrs[attr] = amount
            # make new requirement instances
            new_reqs[reqs] = [req_class(**attrs) for attrs in reqs2attrs[reqs]]
        # update work unit attributes with created requirements
        wu_attrs.update(new_reqs)
        # update id attribute with current stage node id
        wu_attrs['id'] = stage_node_id
        # update name attribute with current index of stage
        wu_attrs['name'] = f'{wu.name}{STAGE_SEP}{stage_i}'
        # update volume attribute with passed proportion
        wu_attrs['volume'] = wu.volume * volume_proportion
        # make new work unit for new stage node with updated attributes
        new_wu = WorkUnit(**wu_attrs)
        # make new graph node for new stage with created work unit and with passed edge to previous stage node
        return GraphNode(new_wu, edge_with_prev_stage_node)

    def match_prev_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id: dict[tuple[str, str], str]):
        """
        Matches the original edges, along which a previous stage node has been created,
        and IDs of the previous and current stage nodes.
        It is needed because the edges entering the node must refer to the next stage node
        for further filling of parent nodes, so matching can be done only after creating the next stage node.
        Edges that leave the node matches to the previous stage node.
        """
        for prev_edge, prev_is_edge_to_node in edges_to_match_with_stage_nodes:
            start, finish = prev_edge.start.id, prev_edge.finish.id
            if prev_is_edge_to_node:
                start, finish = finish, start
            restructuring_edges2new_nodes_id[(start, finish)] = \
                prev_stage_node_id if not prev_is_edge_to_node else stage_node_id

    wu = origin_node.work_unit
    # copy attributes of original work unit
    wu_attrs = dict(wu.__dict__)

    if not restructuring_edges:
        # if there are no edges dividing the node into stages, then simply copy this node
        id2new_nodes[wu.id] = GraphNode(deepcopy(wu), [])
        return

    # define mapper of requirements attribute names and classes
    reqs2classes = {'worker_reqs': WorkerReq, 'equipment_reqs': EquipmentReq,
                    'object_reqs': ConstructionObjectReq, 'material_reqs': MaterialReq}
    # make mapper of work unit requirement names and copied class object attributes
    reqs2attrs = {reqs: [dict(req.__dict__) for req in getattr(wu, reqs)] for reqs in reqs2classes}
    # make mapper of work unit requirement names and accumulated amounts
    reqs2amounts_accum = {reqs: [0 for _ in getattr(wu, reqs)] for reqs in reqs2classes}

    # calculate from the restructuring edges all the proportions of dividing the node into stages
    # depending on the direction and lags of these edges.
    # if an edge comes to a node,
    # then the proportion is calculated as 1 - lag / start.volume if use_lag_edge_optimization else 1
    # if an edge leaves a node,
    # then the proportion is calculated as lag / volume if use_lag_edge_optimization else 0
    proportions_accum = [(int(is_edge_to_node) +
                          ((1 - 2 * int(is_edge_to_node)) * edge.lag / edge.start.work_unit.volume
                           if use_lag_edge_optimization and 0 < edge.lag <= edge.start.work_unit.volume
                           else 0
                           ),
                          edge, is_edge_to_node
                          )
                         for edge, is_edge_to_node in restructuring_edges]
    # sort the resulting proportions in ascending order so that they represent a list of accumulations
    proportions_accum.sort(key=itemgetter(0))
    # initialize index of stages for creating name and id of stage nodes
    stage_i = 0

    # Create first stage node

    # get first proportion with corresponding edge
    accum, edge, is_edge_to_node = proportions_accum[0]
    # get amounts of requirements for the first stage node
    reqs2amounts, reqs2amounts_accum = get_reqs_amounts(accum, reqs2amounts_accum)
    # make id for first stage node
    stage_node_id = make_new_node_id(wu.id, stage_i)
    # make first stage node and add it to id2new_nodes
    id2new_nodes[stage_node_id] = make_new_stage_node(accum, [], wu_attrs, reqs2attrs)

    # initialize a list that stores the edges along which a stage node has already been created.
    # used only if the use_lag_edge_optimization is True,
    # since otherwise the matching of the edges and new node IDs is trivial
    edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)] if use_lag_edge_optimization else None

    # Create intermediate stage nodes

    # iterate over the accumulations following the first, checking the difference with the previous accumulation
    for value, prev_value in zip(proportions_accum[1:], proportions_accum):
        accum, edge, is_edge_to_node = value
        prev_accum, _, _ = prev_value
        if accum == prev_accum:
            # the difference is zero, therefore the corresponding division into the stage has already been made
            if use_lag_edge_optimization:
                # if the use_lag_edge_optimization is True,
                # then add this edge for further matching with IDs of stage nodes
                # when the next stage node will be created
                edges_to_match_with_stage_nodes.append((edge, is_edge_to_node))
            continue
        # increase the stage index for creating name and id of stage node
        stage_i += 1
        # remember the ID of the previous stage node for matching with stored edges
        prev_stage_node_id = stage_node_id
        # make id for new stage node
        stage_node_id = make_new_node_id(wu.id, stage_i)
        # calculate difference between accumulations to obtain the proportion for splitting into the next stage
        proportion = accum - prev_accum
        # get amounts of requirements for the new stage node
        reqs2amounts, reqs2amounts_accum = get_reqs_amounts(proportion, reqs2amounts_accum)
        # make new stage node with InseparableFinishStart edge to previous stage node and add it to id2new_nodes
        id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[prev_stage_node_id], 0,
                                                                        EdgeType.InseparableFinishStart)],
                                                          wu_attrs, reqs2attrs
                                                          )
        if use_lag_edge_optimization:
            # if the use_lag_edge_optimization is True,
            # match stored edges with stage node IDs since a new stage node has been created
            match_prev_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id)
            # store this edge for further matching with IDs of stage nodes
            # when the next stage node will be created
            edges_to_match_with_stage_nodes = [(edge, is_edge_to_node)]

    # Create last stage node

    # increase the stage index for name of last stage node
    stage_i += 1
    # remember the ID of the previous stage node for matching with stored edges
    prev_stage_node_id = stage_node_id
    # ID of the last stage node is equal to the ID of the original node,
    # so that it is more convenient to restore the parent edges
    stage_node_id = wu.id
    # proportion for the last stage is calculated as one minus the last accumulation
    proportion = 1 - accum
    # iterate over the requirements and calculate their amounts for the last stage
    for reqs in reqs2classes:
        # define a name of the amount attribute of the requirement
        attr = 'volume' if reqs == 'worker_reqs' else 'count'
        # amount of requirements for the last stage is calculated
        # as the difference between the amounts of the original node and the accumulated requirement amounts
        reqs2amounts[reqs] = [getattr(req, attr) - req_accum
                              for req, req_accum in zip(getattr(wu, reqs), reqs2amounts_accum[reqs])]
    # make last stage node with InseparableFinishStart edge to previous stage node and add it to id2new_nodes
    id2new_nodes[stage_node_id] = make_new_stage_node(proportion, [(id2new_nodes[prev_stage_node_id], 0,
                                                                    EdgeType.InseparableFinishStart)],
                                                      wu_attrs, reqs2attrs
                                                      )
    if use_lag_edge_optimization:
        # if the use_lag_edge_optimization is True,
        # match stored edges with stage node IDs since the last stage node has been created
        match_prev_edges_with_stage_nodes_id(restructuring_edges2new_nodes_id)


def graph_restructuring(wg: WorkGraph, use_lag_edge_optimization: Optional[bool] = False) -> WorkGraph:
    """Converts a work graph to use only FS and IFS edges.

    Преобразует рабочий граф, оставляя только связи Finish-Start и
    Inseparable-Finish-Start.

    Args:
        wg (WorkGraph): The graph to convert.
            Преобразуемый граф.
        use_lag_edge_optimization (bool, optional): Whether to account for lag
            values on edges. Defaults to ``False``.
            Учитывать ли задержки на рёбрах. По умолчанию ``False``.

    Returns:
        WorkGraph: Restructured work graph.
            WorkGraph: реструктурированный граф.
    """

    def get_restructuring_edges(edges: list[GraphEdge], edge_type: EdgeType, is_edge_to_node: bool) \
            -> list[tuple[GraphEdge, bool]]:
        """
        This function keeping all the edges of the types
        that need to be removed and which will break the node into stages
        and a flag indicating whether the edge enters or leaves the node
        """
        return [(edge, is_edge_to_node) for edge in edges if edge.type is edge_type]

    # mapper of IDs of new nodes and the new nodes themselves that represent the restructured graph
    id2new_nodes: dict[str, GraphNode] = dict()
    # mapper of original edges and IDs of new nodes that should be connected.
    # edge is specified by two IDs of the nodes that this edge connects.
    # the order of the node IDs determines (based on the first ID) on which side of the edge the new node ID is written.
    # used only if the use_lag_edge_optimization is True, since otherwise the connection is trivial
    restructuring_edges2new_nodes_id: dict[tuple[str, str], str] = dict()
    # iterate over each node,
    # keeping all the edges that need to be removed and which will break the node into stages,
    # creating a new node (possibly with stage nodes) that will be in the final restructured graph,
    # and connecting to parent nodes that have already been created since the nodes in wg.nodes are topological sorted
    for node in wg.nodes:
        restructuring_edges = get_restructuring_edges(node.edges_from, EdgeType.StartStart, False) + \
                              get_restructuring_edges(node.edges_to, EdgeType.FinishFinish, True)

        if use_lag_edge_optimization:
            restructuring_edges += get_restructuring_edges(node.edges_to, EdgeType.LagFinishStart, True) + \
                                   get_restructuring_edges(node.edges_from, EdgeType.LagFinishStart, False)

        split_node_into_stages(node, restructuring_edges, id2new_nodes, restructuring_edges2new_nodes_id,
                               use_lag_edge_optimization)
        fill_parents_to_new_nodes(node, id2new_nodes, restructuring_edges2new_nodes_id, use_lag_edge_optimization)

    result = WorkGraph(id2new_nodes[wg.start.id], id2new_nodes[wg.finish.id])
    # TODO Make structurator natively work with priorities
    check_and_correct_priorities(result)
    return result
