from operator import attrgetter
from typing import Dict

from _pytest.fixtures import fixture

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import get_worker_contractor_pool
from sampo.schemas.graph import GraphNode
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.collections_util import build_index


@fixture(scope='function')
def setup_timeline(setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)
    return JustInTimeTimeline(setup_wg.nodes, setup_contractors, setup_worker_pool, landscape=landscape), \
        setup_wg, setup_contractors, setup_worker_pool


def test_schedule(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool = setup_timeline

    ordered_nodes = prioritization(setup_wg, DefaultWorkEstimator())
    node = ordered_nodes[-1]

    reqs = build_index(node.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if
                   name in reqs]

    contractor_index = build_index(setup_contractors, attrgetter('id'))
    contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None

    node2swork: Dict[GraphNode, ScheduledWork] = {}
    setup_timeline.schedule(node, node2swork, worker_team, contractor, WorkSpec())

    assert len(node2swork) == 1
    for swork in node2swork.values():
        assert not swork.finish_time.is_inf()
