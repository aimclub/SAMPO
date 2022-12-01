from operator import attrgetter
from typing import Dict
from uuid import uuid4

from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import ContractorName
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.types import WorkerName
from sampo.utilities.collections import build_index


def test_init_resource_structure(setup_worker_pool):
    timeline = JustInTimeTimeline(setup_worker_pool)
    assert len(timeline._timeline) != 0
    for timeline in timeline._timeline.values():
        assert len(timeline) == 1
        assert timeline[0][0] == 0


def test_update_resource_structure(setup_worker_pool):
    timeline = JustInTimeTimeline(setup_worker_pool)
    mut_name: WorkerName = list(setup_worker_pool.keys())[0]
    mut_contractor: ContractorName = list(setup_worker_pool[mut_name].keys())[0]
    mut_count = timeline[(mut_contractor, mut_name)][0][1]

    # mutate
    worker = Worker(str(uuid4()), mut_name, 1, contractor_id=mut_contractor)
    timeline.update_timeline(Time(1), [worker])

    worker_timeline = timeline[worker.get_agent_id()]

    if mut_count == 1:
        assert len(worker_timeline) == 1
        assert worker_timeline[0] == (Time(0), 1)
    else:
        assert len(worker_timeline) == 2
        assert worker_timeline[0] == (Time(1), 1)
        assert worker_timeline[1] == (Time(0), mut_count - 1)


def test_schedule(setup_wg, setup_worker_pool, setup_contractors):
    timeline = JustInTimeTimeline(setup_worker_pool)
    ordered_nodes = prioritization(setup_wg)
    node = ordered_nodes[-1]

    reqs = build_index(node.work_unit.worker_reqs, attrgetter('kind'))
    worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if name in reqs]

    contractor_index = build_index(setup_contractors, attrgetter('id'))
    contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None

    node2swork: Dict[GraphNode, ScheduledWork] = {}
    ft = timeline.schedule(0, node, node2swork, worker_team, contractor, None)

    assert not ft.is_inf()
    assert len(node2swork) == 1

