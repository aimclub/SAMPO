from pytest import fixture

from sampo.schemas.graph import WorkGraph, EdgeType
from sampo.schemas.requirements import MaterialReq
from sampo.structurator import graph_restructuring
from sampo.structurator.base import make_new_node_id

import numpy as np

pytest_plugins = ("tests.schema", "tests.models",)


@fixture(params=[graph_type for graph_type in ['manual', 'manual with negative lag',
                                               'manual with negative volume', 'manual with lag > volume']],
         ids=[f'Graph: {graph_type}' for graph_type in ['manual', 'manual with negative lag',
                                                        'manual with negative volume', 'manual with lag > volume']]
         )
def setup_wg_for_restructuring(request, setup_sampler, setup_simple_synthetic) -> tuple[WorkGraph, int]:
    sr = setup_sampler

    l1n1 = sr.graph_node('l1n1', [], group='0', work_id='000001')
    l1n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l1n1.work_unit.volume = 50
    l1n2 = sr.graph_node('l1n2', [], group='0', work_id='000002')
    l1n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l1n2.work_unit.volume = 50

    l2n1 = sr.graph_node('l2n1', [(l1n1, 3, EdgeType.LagFinishStart)], group='1', work_id='000011')
    l2n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l2n1.work_unit.volume = 50
    l2n2 = sr.graph_node('l2n2', [(l1n1, 5, EdgeType.LagFinishStart),
                                  (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
    l2n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l2n2.work_unit.volume = 50
    l2n3 = sr.graph_node('l2n3', [(l1n1, 5, EdgeType.StartStart),
                                  (l2n1, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')
    l2n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l2n3.work_unit.volume = 50

    l3n1 = sr.graph_node('l2n1', [(l2n1, 0, EdgeType.FinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
    l3n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l3n1.work_unit.volume = 50
    l3n2 = sr.graph_node('l2n2', [(l2n2, 0, EdgeType.FinishStart),
                                  (l3n1, 5, EdgeType.FinishFinish)], group='2', work_id='000022')
    l3n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l3n2.work_unit.volume = 50
    l3n3 = sr.graph_node('l2n3', [(l2n3, 2, EdgeType.LagFinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')
    l3n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l3n3.work_unit.volume = 50

    wg = WorkGraph.from_nodes([l1n1, l1n2, l2n1, l2n2, l2n3, l3n1, l3n2, l3n3])

    n_nodes_after_restructuring = len(wg.nodes)

    match request.param:
        case 'manual':
            n_nodes_after_restructuring += 9
        case 'manual with negative lag':
            l2n1.add_parents([(l1n2, -1, EdgeType.LagFinishStart)])
            n_nodes_after_restructuring += 11
        case 'manual with negative volume':
            l1n1.work_unit.volume = -50
            n_nodes_after_restructuring += 8
        case 'manual with lag > volume':
            l2n1.add_parents([(l1n2, 60, EdgeType.LagFinishStart)])
            n_nodes_after_restructuring += 11
        case _:
            raise ValueError(f'Unknown graph type: {request.param}')

    return wg, n_nodes_after_restructuring


def test_restructuring(setup_wg_for_restructuring):
    wg_original, n_nodes_after_restructuring = setup_wg_for_restructuring
    wg_restructured = graph_restructuring(wg_original, True)
    assert len(wg_restructured.nodes) == n_nodes_after_restructuring, "Nodes are divided incorrect"
    assert all([edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart]
                for node in wg_restructured.nodes
                for edge in node.edges_to]), \
        "Edges in restructured work graph have not only FS and IFS types"
    wg_nodes_id = [node.id for node in wg_original.nodes]
    wg_restructured_nodes_id = [node.id for node in wg_restructured.nodes]
    assert all([node_id in wg_restructured_nodes_id for node_id in wg_nodes_id]), \
        "Not all nodes from original work graph are in restructured work graph"
    assert all([edge.lag == 0 if edge.start.work_unit.is_service_unit else edge.lag == 1
                for node in wg_restructured.nodes
                for edge in node.edges_to]), \
        "Not all lags in restructured work graph have correct lag amount"
    for node in wg_original.nodes:
        zero_stage = make_new_node_id(node.id, 0)
        if zero_stage in wg_restructured.dict_nodes:
            chain = wg_restructured.dict_nodes[zero_stage].get_inseparable_chain()
            assert chain is not None, f"Node {node.id} - is divided incorrect"
            volume_amount = sum([ins_node.work_unit.volume for ins_node in chain])
            assert volume_amount == node.work_unit.volume, f"Volume of node {node.id} - is divided incorrect"
            workers_volumes = np.array([[req.volume for req in ins_node.work_unit.worker_reqs]
                                        for ins_node in chain])
            assert all(np.array([req.volume for req in node.work_unit.worker_reqs]) == workers_volumes.sum(axis=0)), \
                f"Worker requirements' volumes of node {node.id} - are divided incorrect"
            equipments_counts = np.array([[req.count for req in ins_node.work_unit.equipment_reqs]
                                          for ins_node in chain])
            assert all(np.array([req.count for req in node.work_unit.equipment_reqs]) == equipments_counts.sum(axis=0)
                       ), f"Equipment requirements' counts of node {node.id} - are divided incorrect"
            objects_counts = np.array([[req.count for req in ins_node.work_unit.object_reqs]
                                       for ins_node in chain])
            assert all(np.array([req.count for req in node.work_unit.object_reqs]) == objects_counts.sum(axis=0)
                       ), f"Object requirements' counts of node {node.id} - are divided incorrect"
            materials_counts = np.array([[req.count for req in ins_node.work_unit.material_reqs]
                                         for ins_node in chain])
            assert all(np.array([req.count for req in node.work_unit.material_reqs]) == materials_counts.sum(axis=0)
                       ), f"Material requirements' counts of node {node.id} - are divided incorrect"
