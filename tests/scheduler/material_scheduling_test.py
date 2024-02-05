import uuid

import pytest

from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.timeline import SupplyTimeline
from sampo.schemas import Time, WorkGraph, MaterialReq, EdgeType, LandscapeConfiguration, Material
from sampo.schemas.landscape import Vehicle, ResourceHolder
from sampo.schemas.landscape_graph import LandGraph, ResourceStorageUnit, LandGraphNode
from sampo.utilities.sampler import Sampler
from sampo.utilities.validation import validate_schedule
from tests.conftest import setup_default_schedules


def test_empty_node_find_start_time(setup_default_schedules):
    wg, _, landscape = setup_default_schedules[0]
    if wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    timeline = SupplyTimeline(landscape)
    delivery_time = timeline.find_min_material_time(wg.start, Time(0), wg.start.work_unit.need_materials())

    assert delivery_time == Time(0)
#
#
# def test_ordered_nodes_of_one_platform(setup_default_schedules):
#     wg, _, landscape = setup_default_schedules[0]
#     if wg.vertex_count > 14:
#         pytest.skip('Non-material graph with')
#
#     timeline = SupplyTimeline(landscape)
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#
#     if landscape.lg is None:
#         pytest.skip('there is no landscape')
#     platform = landscape.lg.nodes[0]
#     ordered_nodes = [node for node in ordered_nodes[::-1] if node in platform.works]
#     for node in ordered_nodes:
#         delivery_time = timeline.find_min_material_time(node, landscape, Time(0), node.work_unit.need_materials())
#         assert delivery_time < Time.inf()
#
# def test_ordered_nodes_of_one_platforms_with_schedule(setup_scheduler_parameters):
#     wg, _, landscape = setup_scheduler_parameters
#     if wg.vertex_count > 14:
#         pytest.skip('Non-material graph with')
#     timeline = SupplyTimeline(landscape)
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#
#     if landscape.lg is None:
#         pytest.skip('there is no landscape')
#     platform = [landscape.lg.nodes[i] for i in range(len(landscape.lg.nodes)) if len(landscape.lg.nodes[i].works) > 2][0]
#     ordered_nodes = [node for node in ordered_nodes[::-1] if node in platform.works]
#     for node in ordered_nodes:
#         deliveries, parent_time = timeline.supply_resources(node, landscape, Time(0),
#                                                             node.work_unit.need_materials(), True)
#         assert parent_time < Time.inf()
#
# def test_material_timeline(setup_scheduler_parameters):
#     wg, _, landscape = setup_scheduler_parameters
#     if wg.vertex_count > 14:
#         pytest.skip('Non-material graph with')
#     timeline = SupplyTimeline(landscape)
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#
#     for node in ordered_nodes:
#         deliveries, parent_time = timeline.supply_resources(node, landscape, Time(0),
#                                                             node.work_unit.need_materials(), True)
#         assert parent_time < Time.inf()
#
# def test_just_in_time_scheduling_with_materials(setup_default_schedules):
#     setup_wg, setup_contractors, landscape = setup_default_schedules[0]
#     if setup_wg.vertex_count > 14:
#         pytest.skip('Non-material graph')
#
#     scheduler = HEFTScheduler()
#     schedule = scheduler.schedule(setup_wg, setup_contractors, validate=False, landscape=landscape)
#
#     try:
#         validate_schedule(schedule, setup_wg, setup_contractors)
#
#     except AssertionError as e:
#         raise AssertionError(f'Scheduler {scheduler} failed validation', e)


def test_momentum_scheduling_with_materials(setup_default_schedules):
    setup_wg, setup_contractors, landscape = setup_default_schedules[0]
    if setup_wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    scheduler = HEFTBetweenScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, validate=True, landscape=landscape)

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)

    except AssertionError as e:
        raise AssertionError(f'Scheduler {scheduler} failed validation', e)


def test_scheduler_with_materials_validity_right(setup_schedule):
    schedule = setup_schedule[0]
    setup_wg, setup_contractors, landscape = setup_schedule[2]

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)
    except AssertionError as e:
        raise AssertionError(f'Scheduler {setup_schedule[1]} failed validation', e)


def lg(wg: WorkGraph):
    nodes = wg.nodes
    platform1 = LandGraphNode(str(uuid.uuid4()), 'platform1',
                              ResourceStorageUnit({
                                  'mat1': 200,
                                  'mat2': 150,
                                  'mat3': 120
                              }), works=nodes[1:3])
    platform2 = LandGraphNode(str(uuid.uuid4()), 'platform2',
                              ResourceStorageUnit({
                                  'mat1': 200,
                                  'mat2': 80,
                                  'mat3': 90
                              }), works=nodes[3:5])
    platform3 = LandGraphNode(str(uuid.uuid4()), 'platform3',
                              ResourceStorageUnit({
                                  'mat1': 200,
                                  'mat2': 130,
                                  'mat3': 170
                              }), works=nodes[5:7])
    platform4 = LandGraphNode(str(uuid.uuid4()), 'platform4',
                              ResourceStorageUnit({
                                  'mat1': 200,
                                  'mat2': 190,
                                  'mat3': 200
                              }), works=nodes[7:9])
    holder1 = LandGraphNode(str(uuid.uuid4()), 'holder1',
                            ResourceStorageUnit({
                                'mat1': 12000,
                                'mat2': 500,
                                'mat3': 500
                            }))
    holder2 = LandGraphNode(str(uuid.uuid4()), 'holder2',
                            ResourceStorageUnit({
                                'mat1': 1000,
                                'mat2': 750,
                                'mat3': 800
                            }))
    platform1.add_neighbours([(platform3, 1.0, 10)])
    platform2.add_neighbours([(platform4, 2.0, 10)])
    platform3.add_neighbours([(holder1, 4.0, 10), (holder2, 3.0, 9)])
    platform4.add_neighbours([(holder1, 5.0, 8), (holder2, 7.0, 8)])
    holder1.add_neighbours([(holder2, 6.0, 8)])

    return LandGraph(nodes=[platform1, platform2, platform3, platform4, holder1, holder2]), [holder1, holder2]


def _landscape(lg_info):
    lg, holders = lg_info
    holders = [ResourceHolder(str(uuid.uuid4()), 'holder1',
                              [
                                  Vehicle(str(uuid.uuid4()), 'vehicle1', [
                                      Material('111', 'mat1', 100),
                                      Material('222', 'mat2', 100),
                                      Material('333', 'mat3', 100)
                                  ]),
                                  Vehicle(str(uuid.uuid4()), 'vehicle2', [
                                      Material('111', 'mat1', 150),
                                      Material('222', 'mat2', 150),
                                      Material('333', 'mat3', 150)
                                  ])
                              ],
                              holders[0]),
               ResourceHolder(str(uuid.uuid4()), 'holder2',
                              [
                                  Vehicle(str(uuid.uuid4()), 'vehicle1', [
                                      Material('111', 'mat1', 120),
                                      Material('222', 'mat2', 120),
                                      Material('333', 'mat3', 120)
                                  ]),
                                  Vehicle(str(uuid.uuid4()), 'vehicle2', [
                                      Material('111', 'mat1', 140),
                                      Material('222', 'mat2', 140),
                                      Material('333', 'mat3', 140)
                                  ])
                              ],
                              holders[1])]
    landscape = LandscapeConfiguration(holders=holders, lg=lg)
    return landscape


def _wg():
    sr = Sampler(1e-1)

    l1n1 = sr.graph_node('l1n1', [], group='0', work_id='000001')
    l1n2 = sr.graph_node('l1n2', [], group='0', work_id='000002')

    l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
    l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                                  (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
    l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')

    l3n1 = sr.graph_node('l3n1', [(l2n1, 0, EdgeType.FinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
    l3n2 = sr.graph_node('l3n2', [(l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000022')
    l3n3 = sr.graph_node('l3n3', [(l2n3, 1, EdgeType.LagFinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')


    l1n1.work_unit.material_reqs = [MaterialReq('mat1', 120)]
    l1n2.work_unit.material_reqs = [MaterialReq('mat1', 120)]

    l2n1.work_unit.material_reqs = [MaterialReq('mat1', 120)]
    l2n2.work_unit.material_reqs = [MaterialReq('mat1', 120)]
    l2n3.work_unit.material_reqs = [MaterialReq('mat1', 120)]

    l3n1.work_unit.material_reqs = [MaterialReq('mat1', 120)]
    l3n2.work_unit.material_reqs = [MaterialReq('mat1', 120)]
    l3n3.work_unit.material_reqs = [MaterialReq('mat1', 120)]

    return WorkGraph.from_nodes([l1n1, l1n2, l2n1, l2n2, l2n3, l3n1, l3n2, l3n3])


# def test_schedule_with_intersection():
#     """
#     Here we deal with intersecting of works
#     40-----------------------------23------------->
#                                    S(1)
#     40-----------26-----23---------23------------->
#                  S(2)  S_f(2)      S(1)
#     :return:
#     """
#     wg = _wg()
#     landscape = _landscape(lg(wg))
#     timeline = SupplyTimeline(landscape)
#
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#     platform1_ordered_nodes = [node for node in ordered_nodes if not node.work_unit.is_service_unit
#                                if landscape.works2platform[node].name == 'platform1']
#     platform1_ordered_nodes.reverse()
#
#     node = platform1_ordered_nodes[0]
#     start_time = timeline.find_min_material_time(node, Time(2), node.work_unit.need_materials(), Time(2))
#     assert start_time != Time.inf()
#
#     timeline.deliver_resources(node, start_time, node.work_unit.need_materials(), Time(2), True)
#
#     node2 = platform1_ordered_nodes[1]
#     start_time2 = timeline.find_min_material_time(node2, Time(0), node.work_unit.need_materials(), Time(1))
#     assert start_time2 != Time.inf()
#
#     timeline.deliver_resources(node2, start_time2, node2.work_unit.need_materials(), Time(1), True)
#
#
# def test_schedule_with_intersection_2():
#     """
#     Here we deal with intersecting of works
#     40-----------------------------23------------->
#                                    S(1)
#     40-----------26--------------23-----23------->
#                  S(2)            S(1)   S_f(2)
#     :return:
#     """
#     wg = _wg()
#     landscape = _landscape(lg(wg))
#     timeline = SupplyTimeline(landscape)
#
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#     platform1_ordered_nodes = [node for node in ordered_nodes if not node.work_unit.is_service_unit
#                                if landscape.works2platform[node].name == 'platform1']
#     platform1_ordered_nodes.reverse()
#
#     node = platform1_ordered_nodes[0]
#     start_time = timeline.find_min_material_time(node, Time(2), node.work_unit.need_materials(), Time(2))
#     assert start_time != Time.inf()
#
#     timeline.deliver_resources(node, start_time, node.work_unit.need_materials(), Time(2), True)
#
#     node2 = platform1_ordered_nodes[1]
#     start_time2 = timeline.find_min_material_time(node2, Time(0), node.work_unit.need_materials(), Time(4))
#     assert start_time2 != Time.inf()
#
#     timeline.deliver_resources(node2, start_time2, node2.work_unit.need_materials(), Time(4), True)
#
#
# def test_schedule_with_intersection_3():
#     """
#     Here we deal with intersecting of works. Start time of S(2) replaced after S(1)
#     40-----------------------------23------------->
#                                    S(1)
#     40-----------26--------------23-----23------->
#                  S(2)            S(1)   S_f(2)
#     :return:
#     """
#     wg = _wg()
#     landscape = _landscape(lg(wg))
#     timeline = SupplyTimeline(landscape)
#
#     ordered_nodes = prioritization(wg, DefaultWorkEstimator())
#     platform1_ordered_nodes = [node for node in ordered_nodes if not node.work_unit.is_service_unit
#                                if landscape.works2platform[node].name == 'platform1']
#     platform1_ordered_nodes.reverse()
#
#     node = platform1_ordered_nodes[0]
#     start_time = timeline.find_min_material_time(node, Time(2), node.work_unit.need_materials(), Time(2))
#     assert start_time != Time.inf()
#
#     timeline.deliver_resources(node, start_time, node.work_unit.need_materials(), Time(2), True)
#
#     node2 = platform1_ordered_nodes[1]
#     start_time2 = timeline.find_min_material_time(node2, Time(1), node.work_unit.need_materials(), Time(2))
#     # assert start_time2 > Time(3)
#
#     timeline.deliver_resources(node2, start_time2, node2.work_unit.need_materials(), Time(2), True)