from copy import deepcopy

from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit


def copy_graph_node(node: GraphNode, new_id: int | str | None = None,
                    change_id: bool = True) -> tuple[GraphNode, tuple[str, str]]:
    """
    Makes a deep copy of GraphNode without edges. It's id can be changed to a new randomly generated or specified one
    :param node: original GraphNode
    :param new_id: specified new id
    :param change_id: do ids in the new graph need to be changed
    :return: copy of GraphNode and pair(old node id, new node id)
    """
    if change_id:
        new_id = new_id or uuid_str()
        new_id = str(new_id) if isinstance(new_id, int) else new_id
    else:
        new_id = node.work_unit.id
    wu = node.work_unit
    new_wu = WorkUnit(id=new_id, name=wu.name,
                      worker_reqs=deepcopy(wu.worker_reqs),
                      material_reqs=deepcopy(wu.material_reqs),
                      equipment_reqs=deepcopy(wu.equipment_reqs),
                      object_reqs=deepcopy(wu.object_reqs),
                      zone_reqs=deepcopy(wu.zone_reqs),
                      group=wu.group,
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


def prepare_work_graph_copy(wg: WorkGraph, excluded_nodes: list[GraphNode] = [], use_ids_simplification: bool = False,
                            id_offset: int = 0, change_id: bool = True) -> tuple[dict[str, GraphNode], dict[str, str]]:
    """
    Makes a deep copy of the GraphNodes of the original graph with new ids and updated edges,
    ignores all GraphNodes specified in the exception list and GraphEdges associated with them
    :param wg: original WorkGraph for copy
    :param excluded_nodes: GraphNodes to be excluded from the graph
    :param use_ids_simplification: If true, creates short numeric ids converted to strings,
    otherwise uses uuid to generate id
    :param id_offset: Shift for numeric ids, used only if param use_ids_simplification is True
    :param change_id: Do IDs in the new graph need to be changed
    :return: A dictionary with GraphNodes by their id
    and a dictionary linking the ids of GraphNodes of the original graph and the new GraphNode ids
    """
    excluded_nodes = {node.id for node in excluded_nodes}
    node_list = [(id_offset + ind, node) for ind, node in enumerate(wg.nodes)] \
        if use_ids_simplification \
        else [(None, node) for node in wg.nodes]
    nodes, old_to_new_ids = list(zip(*[copy_graph_node(node, ind, change_id) for ind, node in node_list
                                       if node.id not in excluded_nodes]))
    id_old_to_new = dict(old_to_new_ids)
    nodes = {node.id: node for node in nodes}
    restore_parents(nodes, wg, id_old_to_new, excluded_nodes)
    return nodes, id_old_to_new


def new_start_finish(original_wg: WorkGraph, copied_nodes: dict[str, GraphNode],
                     old_to_new_ids: dict[str, str]) -> (GraphNode, GraphNode):
    """
    Prepares new start and finish to create WorkGraph after copying it
    :param original_wg: WorkGraph, on which base prepare_work_graph_copy was run
    :param copied_nodes: New nodes, on which to create new WorkGraph
    :param old_to_new_ids: Dictionary to translate old nodes to new, using their IDs
    :return: new start and new finish nodes, on the base of which to create a WorkGraph
    """
    new_start = copied_nodes[old_to_new_ids[original_wg.start.id]]
    new_finish = copied_nodes[old_to_new_ids[original_wg.finish.id]]
    return new_start, new_finish
