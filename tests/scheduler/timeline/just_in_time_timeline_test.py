from operator import attrgetter
from typing import Dict
from uuid import uuid4

from _pytest.fixtures import fixture

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import ContractorName, get_worker_contractor_pool
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.types import WorkerName
from sampo.utilities.collections_util import build_index
from schemas.time_estimator import AbstractWorkEstimator


@fixture(scope='function')
def setup_timeline(setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)
    return JustInTimeTimeline(setup_wg.nodes, setup_contractors, setup_worker_pool, landscape=landscape), \
        setup_wg, setup_contractors, setup_worker_pool


def test_init_resource_structure(setup_timeline):
    setup_timeline, _, _, _ = setup_timeline

    assert len(setup_timeline._timeline) != 0
    for setup_timeline in setup_timeline._timeline.values():
        assert len(setup_timeline) == 1
        assert setup_timeline[0][0] == 0


def test_update_resource_structure(setup_timeline):
    setup_timeline, _, _, setup_worker_pool = setup_timeline

    mut_name: WorkerName = list(setup_worker_pool.keys())[0]
    mut_contractor: ContractorName = list(setup_worker_pool[mut_name].keys())[0]
    mut_count = setup_timeline[(mut_contractor, mut_name)][0][1]

    # mutate
    worker = Worker(str(uuid4()), mut_name, 1, contractor_id=mut_contractor)
    setup_timeline.update_timeline(Time(1), None, {}, [worker])

    worker_timeline = setup_timeline[worker.get_agent_id()]

    if mut_count == 1:
        assert len(worker_timeline) == 1
        assert worker_timeline[0] == (Time(0), 1)
    else:
        assert len(worker_timeline) == 2
        assert worker_timeline[0] == (Time(2), 1)
        assert worker_timeline[1] == (Time(0), mut_count - 1)


def test_schedule(setup_timeline):
    setup_timeline, setup_wg, setup_contractors, setup_worker_pool = setup_timeline

    ordered_nodes = prioritization(setup_wg, AbstractWorkEstimator())
    node = ordered_nodes[-1]

    reqs = build_index(node.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if name in reqs]

    contractor_index = build_index(setup_contractors, attrgetter('id'))
    contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None

    node2swork: Dict[GraphNode, ScheduledWork] = {}
    setup_timeline.schedule(node, node2swork, worker_team, contractor)

    assert len(node2swork) == 1
    for swork in node2swork.values():
        assert not swork.finish_time.is_inf()

