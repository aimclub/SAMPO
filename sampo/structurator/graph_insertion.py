from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy, new_start_finish


def graph_in_graph_insertion(master_wg: WorkGraph, master_start: GraphNode, master_finish: GraphNode,
                             slave_wg: WorkGraph, change_id: bool = True) -> WorkGraph:
    """
    Inserts the slave WorkGraph into the master WorkGraph,
    while the starting vertex slave_wg becomes the specified master_start,
    and the finishing vertex is correspondingly master_finish
    :param master_wg: the WorkGraph into which the insertion is performed
    :param master_start: GraphNode which will become the parent for the entire slave_wg
    :param master_finish: GraphNode which will become a child for the whole slave_wg
    :param slave_wg: WorkGraph to be inserted into master_wg
    :param change_id: do ids in the new graph need to be changed
    :return: new union WorkGraph
    """
    master_nodes, master_old_to_new_ids = prepare_work_graph_copy(master_wg, change_id=change_id)
    slave_nodes, slave_old_to_new = prepare_work_graph_copy(slave_wg, [slave_wg.start, slave_wg.finish], change_id)

    # add parent links from slave_wg's nodes to master_start node from slave_wg's nodes
    master_start = master_nodes[master_old_to_new_ids[master_start.id]]
    for edge in slave_wg.start.edges_from:
        slave_nodes[slave_old_to_new[edge.finish.id]].add_parents([master_start])

    # add parent links from master_finish to slave_wg's nodes
    master_finish = master_nodes[master_old_to_new_ids[master_finish.id]]
    master_finish.add_parents([slave_nodes[slave_old_to_new[edge.start.id]]
                               for edge in slave_wg.finish.edges_to])

    start, finish = new_start_finish(master_wg, master_nodes, master_old_to_new_ids)
    return WorkGraph(start, finish)
