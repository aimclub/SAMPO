import pytest

from sampo.scheduler.heft.base import HEFTBetweenScheduler, HEFTScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline import SupplyTimeline
from sampo.schemas import Time
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.validation import validate_schedule
from tests.conftest import setup_default_schedules

# def test_empty_node_find_start_time(setup_default_schedules):
#     wg, _, landscape = setup_default_schedules[0]
#     if wg.vertex_count > 14:
#         pytest.skip('Non-material graph')
#
#     timeline = SupplyTimeline(landscape)
#     delivery_time = timeline.find_min_material_time(wg.start, landscape, Time(0), wg.start.work_unit.need_materials())
#
#     assert delivery_time == Time(0)
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
