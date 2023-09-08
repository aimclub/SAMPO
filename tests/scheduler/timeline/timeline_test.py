from operator import attrgetter
from typing import Dict
from uuid import uuid4

import pytest
from _pytest.fixtures import fixture

from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.schemas.contractor import ContractorName, get_worker_contractor_pool
from sampo.schemas.exceptions import NoAvailableResourcesError
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.schemas.types import WorkerName
from sampo.utilities.collections_util import build_index
from sampo.utilities.validation import validate_schedule


@fixture(scope='session',
         params=[JustInTimeTimeline, MomentumTimeline],
         ids=[f'Timeline: {timeline}' for timeline in ['JustInTimeTimeline', 'MomentumTimeline']])
def setup_timeline_type(request):
    return request.param


@fixture
def setup_timeline(setup_timeline_type, setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)
    worker_kinds = set([w_kind for contractor in setup_contractors for w_kind in contractor.workers.keys()])
    return setup_timeline_type(setup_wg.nodes, setup_contractors, setup_worker_pool, landscape=landscape), \
        setup_wg, setup_contractors, setup_worker_pool, worker_kinds, landscape


def match_timeline_to_scheduler(timeline):
    if isinstance(timeline, JustInTimeTimeline):
        return HEFTScheduler
    if isinstance(timeline, MomentumTimeline):
        return HEFTBetweenScheduler


def test_find_min_start_time(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool, _, _ = setup_timeline

    ordered_nodes = prioritization(setup_wg, DefaultWorkEstimator())
    parent = ordered_nodes[1].edges_to[0].start
    child = ordered_nodes[1]
    edge = ordered_nodes[1].edges_to[0]

    reqs_parent = build_index(parent.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if
                   name in reqs_parent]

    contractor_index = build_index(setup_contractors, attrgetter('id'))
    contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None

    node2swork: Dict[GraphNode, ScheduledWork] = {}
    parent_start_time, parent_finish_time, _ = \
        setup_timeline.find_min_start_time_with_additional(parent, worker_team, node2swork, WorkSpec())
    setup_timeline.schedule(parent, node2swork, worker_team, contractor, WorkSpec())

    reqs_child = build_index(child.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if
                   name in reqs_child]

    node2swork = {}
    child_start_time, child_finish_time, _ = \
        setup_timeline.find_min_start_time_with_additional(child, worker_team, node2swork, WorkSpec(),
                                                           parent_finish_time + 1)

    if edge.type.value == 'FS':
        assert child_start_time >= parent_finish_time
    elif edge.type.value == 'FFS':
        assert child_start_time + edge.lag >= parent_finish_time


def test_update_timeline(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool, worker_kinds, _ = setup_timeline

    node = setup_wg.finish.parents[0]

    worker_for_update: WorkerName = list(setup_worker_pool.keys())[0]
    contractor_for_update: ContractorName = list(setup_worker_pool[worker_for_update].keys())[0]
    worker = Worker(str(uuid4()), worker_for_update, 1, contractor_id=contractor_for_update)

    node2swork = {
        node: ScheduledWork(work_unit=node.work_unit,
                            start_end_time=(Time(0), Time(1)),
                            workers=[worker],
                            contractor=contractor_for_update)
    }

    setup_timeline.update_timeline(Time(1), node, node2swork, [worker], WorkSpec())

    worker = worker.copy()
    worker.count = (setup_worker_pool[worker_for_update][contractor_for_update].count - worker.count)
    node2swork[node] = ScheduledWork(work_unit=node.work_unit,
                                     start_end_time=(Time(0), Time(1)),
                                     workers=[worker],
                                     contractor=contractor_for_update)

    setup_timeline.update_timeline(Time(1), node, node2swork, [worker], WorkSpec())


@pytest.mark.xfail(raises=NoAvailableResourcesError,
                   reason='Test for validate resource update timeline failing')
def test_update_timeline_fail(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool, worker_kinds, _ = setup_timeline

    node = setup_wg.finish.parents[0]

    worker_for_update: WorkerName = list(setup_worker_pool.keys())[0]
    contractor_for_update: ContractorName = list(setup_worker_pool[worker_for_update].keys())[0]
    worker = Worker(str(uuid4()), worker_for_update, 1, contractor_id=contractor_for_update)

    node2swork = {
        node: ScheduledWork(work_unit=node.work_unit,
                            start_end_time=(Time(0), Time(1)),
                            workers=[worker],
                            contractor=contractor_for_update)
    }

    setup_timeline.update_timeline(Time(1), node, node2swork, [worker], WorkSpec())

    worker = worker.copy()
    worker.count = (setup_worker_pool[worker_for_update][contractor_for_update].count + 1)
    node2swork[node] = ScheduledWork(work_unit=node.work_unit,
                                     start_end_time=(Time(0), Time(1)),
                                     workers=[worker],
                                     contractor=contractor_for_update)

    setup_timeline.update_timeline(Time(1), node, node2swork, [worker], WorkSpec())


def test_schedule(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool, worker_kinds, _ = setup_timeline

    node = setup_wg.finish.parents[0]

    reqs = build_index(node.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if
                   name in reqs]

    contractor_index = build_index(setup_contractors, attrgetter('id'))
    contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None
    worker = Worker(str(uuid4()), worker_team[0].name, 1, contractor_id=contractor.id)

    node2swork = {
        node: ScheduledWork(work_unit=node.work_unit,
                            start_end_time=(Time(0), Time(1)),
                            workers=[worker],
                            contractor=contractor.id)
    }
    setup_timeline.schedule(node, node2swork, worker_team, contractor, WorkSpec())

    assert len(node2swork) == 1
    for swork in node2swork.values():
        assert not swork.finish_time.is_inf()


def test_timeline_scheduling_with_materials(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool, worker_kinds, landscape = setup_timeline
    if setup_wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    scheduler = match_timeline_to_scheduler(setup_timeline)()
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

