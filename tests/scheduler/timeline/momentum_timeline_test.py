from _pytest.fixtures import fixture

from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.schemas.graph import GraphNode
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.time import Time
from sampo.schemas.types import ScheduleEvent, EventType
from sampo.schemas.works import WorkUnit


@fixture
def setup_timeline_context(setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape, spec, rand = setup_scheduler_parameters
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)
    worker_kinds = set([w_kind for contractor in setup_contractors for w_kind in contractor.workers.keys()])
    return MomentumTimeline(setup_worker_pool, landscape=landscape), \
        setup_wg, setup_contractors, spec, rand, setup_worker_pool, worker_kinds


def test_init_resource_structure(setup_timeline_context):
    timeline, wg, contractors, _, _, worker_pool, worker_kinds = setup_timeline_context
    assert len(timeline._timeline) != 0

    for contractor_timeline in timeline._timeline.values():
        assert len(contractor_timeline) == len(worker_kinds)

        for worker_timeline in contractor_timeline.values():
            assert len(worker_timeline) == 1

            first_event: ScheduleEvent = worker_timeline[0]
            assert first_event.seq_id == -1
            assert first_event.event_type == EventType.INITIAL
            assert first_event.time == Time(0)


def test_insert_works_with_one_worker_kind(setup_timeline_context):
    timeline, wg, contractors, _, _, worker_pool, worker_kinds = setup_timeline_context

    worker_kind = worker_kinds.pop()
    worker_kinds.add(worker_kind)  # make worker_kinds stay unchanged

    nodes = []

    for i in range(10):
        work_unit = WorkUnit(id=str(i), name=f'Work {str(i)}', worker_reqs=[WorkerReq(kind=worker_kind,
                                                                                      volume=Time(50),
                                                                                      min_count=10,
                                                                                      max_count=50)])
        nodes.append(GraphNode(work_unit=work_unit, parent_works=[]))

    node2swork = {}
    contractor = contractors[0]
    worker_count = contractor.workers[worker_kind].count
    for i, node in enumerate(nodes):
        worker_team = [Worker(id=str(i), name=worker_kind, count=worker_count // 2, contractor_id=contractor.id)]
        timeline.schedule(node, node2swork, worker_team, contractor, WorkSpec())
#
# TODO
# def test_update_resource_structure(setup_timeline, setup_worker_pool):
#     mut_name: WorkerName = list(setup_worker_pool.keys())[0]
#     mut_contractor: ContractorName = list(setup_worker_pool[mut_name].keys())[0]
#     mut_count = setup_timeline[(mut_contractor, mut_name)][0][1]
#
#     # mutate
#     worker = Worker(str(uuid4()), mut_name, 1, contractor_id=mut_contractor)
#     setup_timeline.update_timeline(0, Time(1), None, {}, [worker])
#
#     worker_timeline = setup_timeline[worker.get_agent_id()]
#
#     if mut_count == 1:
#         assert len(worker_timeline) == 1
#         assert worker_timeline[0] == (Time(0), 1)
#     else:
#         assert len(worker_timeline) == 2
#         assert worker_timeline[0] == (Time(1), 1)
#         assert worker_timeline[1] == (Time(0), mut_count - 1)
#
# TODO
# def test_schedule(setup_wg, setup_worker_pool, setup_contractors, setup_timeline):
#     ordered_nodes = prioritization(setup_wg)
#     node = ordered_nodes[-1]
#
#     reqs = build_index(node.work_unit.worker_reqs, attrgetter('kind'))
#     worker_team = [list(cont2worker.values())[0].copy() for name, cont2worker in setup_worker_pool.items() if
#                    name in reqs]
#
#     contractor_index = build_index(setup_contractors, attrgetter('id'))
#     contractor = contractor_index[worker_team[0].contractor_id] if worker_team else None
#
#     node2swork: Dict[GraphNode, ScheduledWork] = {}
#     setup_timeline.schedule(0, node, node2swork, worker_team, contractor, work_estimator=None)
#
#     assert len(node2swork) == 1
#     for swork in node2swork.values():
#         assert not swork.finish_time.is_inf()
