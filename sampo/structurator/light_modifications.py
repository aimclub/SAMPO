from sampo.schemas.graph import WorkGraph
from sampo.structurator.graph_insertion import prepare_work_graph_copy
from sampo.structurator.prepare_wg_copy import new_start_finish


def work_graph_ids_simplification(wg: WorkGraph, id_offset: int = 0, change_id: bool = True) -> WorkGraph:
    """
    Creates a new WorkGraph with simplified numeric ids (numeric ids are converted to a string)
    :param wg: original WorkGraph
    :param id_offset: start for numbering new ids
    :param change_id: Do IDs in the new graph need to be changed
    :return: new WorkGraph with numeric ids
    """
    nodes, old_to_new_ids = prepare_work_graph_copy(wg, [], use_ids_simplification=True, id_offset=id_offset)
    start = nodes[old_to_new_ids[wg.start.id]]
    finish = nodes[old_to_new_ids[wg.finish.id]]
    return WorkGraph(start, finish)
