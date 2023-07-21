from pytest import fixture

from sampo.generator.pipeline.project import get_start_stage, get_finish_stage
from sampo.schemas.graph import WorkGraph, EdgeType
from sampo.schemas.requirements import MaterialReq
from sampo.structurator import graph_restructuring

pytest_plugins = ("tests.schema", "tests.models",)


@fixture(params=[graph_type for graph_type in ['manual', 'small plain synthetic', 'big plain synthetic']],
         # 'small advanced synthetic', 'big advanced synthetic']],
         ids=[f'Graph: {graph_type}'
              for graph_type in ['manual', 'small plain synthetic', 'big plain synthetic']])
# 'small advanced synthetic', 'big advanced synthetic']])
def setup_wg_for_restructuring(request, setup_sampler, setup_simple_synthetic) -> WorkGraph:
    SMALL_GRAPH_SIZE = 100
    BIG_GRAPH_SIZE = 300
    BORDER_RADIUS = 20
    ADV_GRAPH_UNIQ_WORKS_PROP = 0.4
    ADV_GRAPH_UNIQ_RES_PROP = 0.2

    match request.param:
        case 'manual':
            sr = setup_sampler
            s = get_start_stage()

            l1n1 = sr.graph_node('l1n1', [(s, 0, EdgeType.FinishStart)], group='0', work_id='000001')
            l1n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
            l1n2 = sr.graph_node('l1n2', [(s, 0, EdgeType.FinishStart)], group='0', work_id='000002')
            l1n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]

            l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
            l2n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
            l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                                          (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
            l2n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
            l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart),
                                          (l2n1, 5, EdgeType.StartStart)], group='1', work_id='000013')
            l2n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

            l3n1 = sr.graph_node('l2n1', [(l2n1, 0, EdgeType.FinishStart),
                                          (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
            l3n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
            l3n2 = sr.graph_node('l2n2', [(l2n2, 0, EdgeType.FinishStart),
                                          (l3n1, 5, EdgeType.FinishFinish)], group='2', work_id='000022')
            l3n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
            l3n3 = sr.graph_node('l2n3', [(l2n3, 1, EdgeType.LagFinishStart),
                                          (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')
            l3n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

            f = get_finish_stage([l3n1, l3n2, l3n3])
            wg = WorkGraph(s, f)
        case 'small plain synthetic':
            wg = setup_simple_synthetic.work_graph(bottom_border=SMALL_GRAPH_SIZE - BORDER_RADIUS,
                                                   top_border=SMALL_GRAPH_SIZE + BORDER_RADIUS)
        case 'big plain synthetic':
            wg = setup_simple_synthetic.work_graph(bottom_border=BIG_GRAPH_SIZE - BORDER_RADIUS,
                                                   top_border=BIG_GRAPH_SIZE + BORDER_RADIUS)
        case 'small advanced synthetic':
            size = SMALL_GRAPH_SIZE + BORDER_RADIUS
            wg = setup_simple_synthetic.advanced_work_graph(works_count_top_border=size,
                                                            uniq_works=int(size * ADV_GRAPH_UNIQ_WORKS_PROP),
                                                            uniq_resources=int(size * ADV_GRAPH_UNIQ_RES_PROP))
        case 'big advanced synthetic':
            size = BIG_GRAPH_SIZE + BORDER_RADIUS
            wg = setup_simple_synthetic.advanced_work_graph(works_count_top_border=size,
                                                            uniq_works=int(size * ADV_GRAPH_UNIQ_WORKS_PROP),
                                                            uniq_resources=int(size * ADV_GRAPH_UNIQ_RES_PROP))
        case _:
            raise ValueError(f'Unknown graph type: {request.param}')

    return wg


def test_restructuring(setup_wg_for_restructuring):
    wg_restructured = graph_restructuring(setup_wg_for_restructuring, True)
    assert all([edge.type in [EdgeType.FinishStart, EdgeType.InseparableFinishStart] and not edge.lag
                for node in wg_restructured.nodes
                for edge in node.edges_to]), \
        "Edges in restructured work graph have not only FS and IFS types or not zero lag"
    wg_nodes_id = [node.id for node in setup_wg_for_restructuring.nodes]
    wg_restructured_nodes_id = [node.id for node in wg_restructured.nodes]
    assert all([node_id in wg_restructured_nodes_id for node_id in wg_nodes_id]), \
        "Not all nodes from original work graph are in restructured work graph"
