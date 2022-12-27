from copy import deepcopy

from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit


def copy_graph_node(node: GraphNode, new_id: int | str | None = None) -> tuple[GraphNode, tuple[str, str]]:
    """
    Makes a deep copy of GraphNode without edges, with the id changed to a new randomly generated or specified one
    :param node: original GraphNode
    :param new_id: specified new id
    :return: the copy of GraphNode and pair(old node id, new node id)
    """
    new_id = new_id or uuid_str()
    new_id = str(new_id) if isinstance(new_id, int) else new_id
    wu = node.work_unit
    new_wu = WorkUnit(id=new_id, name=wu.name, worker_reqs=deepcopy(wu.worker_reqs), group=wu.group,
                      is_service_unit=wu.is_service_unit, volume=wu.volume, volume_type=wu.volume_type)
    return GraphNode(new_wu, []), (wu.id, new_id)


def restore_parents(new_nodes: dict[str, GraphNode], original_wg: WorkGraph, old_to_new_ids: dict[str, str],
                    excluded_ids: set[str]) -> None:
    """
    Restores edges in GraphNode for copied WorkGraph with changed ids
    :param new_nodes: needed copied nodes
    :param original_wg: original WorkGraph for edge restoring for new nodes
    :param excluded_ids: dictionary of relationships between old ids and new ids
    :param old_to_new_ids: a dictionary linking the ids of GraphNodes of the original graph and the new GraphNode ids
    :return:
    """
    for node in original_wg.nodes:
        if node.id in old_to_new_ids and node.id not in excluded_ids:
            new_node = new_nodes[old_to_new_ids[node.id]]
            new_node.add_parents([(new_nodes[old_to_new_ids[edge.start.id]], edge.lag, edge.type)
                                  for edge in node.edges_to
                                  if edge.start.id in old_to_new_ids and edge.start.id not in excluded_ids])


def prepare_work_graph_copy(wg: WorkGraph, excluded_nodes: list[GraphNode], use_ids_simplification: bool = False,
                            id_offset: int = 0) -> (dict[str, GraphNode], dict[str, str]):
    """
    Makes a deep copy of the GraphNodes of the original graph with new ids and updated edges,
    ignores all GraphNodes specified in the exception list and GraphEdges associated with them
    :param wg: original WorkGraph for copy
    :param excluded_nodes: GraphNodes to be excluded from the graph
    :param use_ids_simplification: If true, creates short numeric ids converted to strings,
    otherwise uses uuid to generate id
    :param id_offset: shift for numeric ids, used only if param use_ids_simplification is True
    :return: a dictionary with GraphNodes by their id
    and a dictionary linking the ids of GraphNodes of the original graph and the new GraphNode ids
    """
    excluded_nodes = {node.id for node in excluded_nodes}
    node_list = [(id_offset + ind, node) for ind, node in enumerate(wg.nodes)] \
        if use_ids_simplification \
        else [(None, node) for node in wg.nodes]
    nodes, old_to_new_ids = list(zip(*[copy_graph_node(node, ind) for ind, node in node_list
                                       if node.id not in excluded_nodes]))
    id_old_to_new = dict(old_to_new_ids)
    nodes = {node.id: node for node in nodes}
    restore_parents(nodes, wg, id_old_to_new, excluded_nodes)
    return nodes, id_old_to_new


def graph_in_graph_insertion(master_wg: WorkGraph, master_start: GraphNode, master_finish: GraphNode,
                             slave_wg: WorkGraph) -> WorkGraph:
    """
    Inserts the slave Work Graph into the masterWork Graph,
    while the starting vertex slave_wg becomes the specified master_start,
    and the finishing vertex is correspondingly master_finish
    :param master_wg: The WorkGraph into which the insertion is performed
    :param master_start: GraphNode which will become the parent for the entire slave_wg
    :param master_finish: GraphNode which will become a child for the whole slave_wg
    :param slave_wg: WorkGraph to be inserted into master_wg
    :return: new union WorkGraph
    """
    master_nodes, master_old_to_new_ids = prepare_work_graph_copy(master_wg, [])
    slave_nodes, slave_old_to_new = prepare_work_graph_copy(slave_wg, [slave_wg.start, slave_wg.finish])

    # add parent links from slave_wg's nodes to master_start node from slave_wg's nodes
    master_start = master_nodes[master_old_to_new_ids[master_start.id]]
    for edge in slave_wg.start.edges_from:
        slave_nodes[slave_old_to_new[edge.finish.id]].add_parents([master_start])

    # add parent links from master_finish to slave_wg's nodes
    master_finish = master_nodes[master_old_to_new_ids[master_finish.id]]
    master_finish.add_parents([slave_nodes[slave_old_to_new[edge.start.id]]
                               for edge in slave_wg.finish.edges_to])

    start = master_nodes[master_old_to_new_ids[master_wg.start.id]]
    finish = master_nodes[master_old_to_new_ids[master_wg.finish.id]]
    return WorkGraph(start, finish)
