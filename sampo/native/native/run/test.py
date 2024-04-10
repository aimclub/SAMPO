import sampo.scheduler

from sampo.backend.native import NativeComputationalBackend
from sampo.base import SAMPO
from sampo.generator.environment import get_contractor_by_wg
from sampo.schemas import MaterialReq, EdgeType, WorkGraph, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.utilities.sampler import Sampler

sr = Sampler()

l1n1 = sr.graph_node('l1n1', [], group='0', work_id='000001')
l1n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
l1n2 = sr.graph_node('l1n2', [], group='0', work_id='000002')
l1n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]

l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
l2n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                              (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
l2n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')
l2n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

l3n1 = sr.graph_node('l3n1', [(l2n1, 0, EdgeType.FinishStart),
                              (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
l3n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
l3n2 = sr.graph_node('l3n2', [(l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000022')
l3n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
l3n3 = sr.graph_node('l3n3', [(l2n3, 1, EdgeType.LagFinishStart),
                              (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')
l3n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

wg = WorkGraph.from_nodes([l1n1, l1n2, l2n1, l2n2, l2n3, l3n1, l3n2, l3n3])

# wg.dump('.', 'wg')

contractors = [get_contractor_by_wg(wg)]

SAMPO.backend = NativeComputationalBackend()

from native import ddd

ddd(wg)

SAMPO.backend.cache_scheduler_info(wg, contractors, LandscapeConfiguration(), ScheduleSpec())
