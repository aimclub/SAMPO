from collections import defaultdict
from typing import Dict, List, Optional
from uuid import uuid4

from pytest import fixture

from sampo.generator.pipeline.project import get_start_stage, get_finish_stage
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.schemas.contractor import WorkerContractorPool, Contractor, DefaultContractorCapacity
from sampo.schemas.graph import WorkGraph, EdgeType
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.structurator.base import graph_restructuring
from sampo.utilities.generation.work_graph import generate_resources_pool
from sampo.utilities.sampler import Sampler

pytest_plugins = ("tests.schema", "tests.models",)


@fixture(scope='session')
def setup_sampler(request):
    return Sampler(1e-1)


@fixture(scope='function')
def setup_wg(request, setup_sampler):
    sr = setup_sampler
    s = get_start_stage()

    l1n1 = sr.graph_node('l1n1', [(s, 0, EdgeType.FinishStart)], group='0', work_id='000001')
    l1n2 = sr.graph_node('l1n2', [(s, 0, EdgeType.FinishStart)], group='0', work_id='000002')

    l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
    l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                                  (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
    l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')

    l3n1 = sr.graph_node('l2n1', [(l2n1, 0, EdgeType.FinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
    l3n2 = sr.graph_node('l2n2', [(l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000022')
    l3n3 = sr.graph_node('l2n3', [(l2n3, 1, EdgeType.LagFinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')

    f = get_finish_stage([l3n1, l3n2, l3n3])
    wg = WorkGraph(s, f)
    wg_r = graph_restructuring(wg, use_lag_edge_optimization=True)
    return wg_r


@fixture(scope='function')
def setup_worker_pool(setup_contractors) -> WorkerContractorPool:
    worker_pool = defaultdict(dict)
    for contractor in setup_contractors:
        for worker in contractor.workers.values():
            worker_pool[worker.name][worker.contractor_id] = worker
    return worker_pool


@fixture(scope='function',
         params=[(i, 5 * j) for j in range(10) for i in range(1, 6)],
         ids=[f'Contractors: count={i}, min_size={5 * j}' for j in range(10) for i in range(1, 6)])
def setup_contractors(request, setup_wg) -> List[Contractor]:
    resource_req: Dict[str, int] = {}
    resource_req_count: Dict[str, int] = {}

    num_contractors, contractor_min_resources = request.param

    for node in setup_wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_req[req.kind] = max(contractor_min_resources,
                                         resource_req.get(req.kind, 0) + (req.min_count + req.max_count) // 2)
            resource_req_count[req.kind] = resource_req_count.get(req.kind, 0) + 1

    for req in resource_req.keys():
        resource_req[req] //= resource_req_count[req]

    # contractors are the same and universal(but multiple)
    contractors = []
    for i in range(num_contractors):
        contractor_id = str(uuid4())
        contractors.append(Contractor(id=contractor_id,
                                      name="OOO Berezka",
                                      workers={name:
                                                   Worker(str(uuid4()), name, count, contractor_id=contractor_id)
                                               for name, count in resource_req.items()},
                                      equipments={}))
    print(f'Using {num_contractors} contractors:')
    print(contractors)
    return contractors


@fixture(scope='function')
def setup_default_schedules(setup_wg, setup_contractors):
    work_estimator: Optional[WorkTimeEstimator] = None

    print(f'Received {len(setup_contractors)} contractors: {setup_contractors}')

    def init_schedule(scheduler_class):
        return scheduler_class(work_estimator=work_estimator).schedule(setup_wg, setup_contractors)

    return {
        "heft_end": init_schedule(HEFTScheduler),
        "heft_between": init_schedule(HEFTBetweenScheduler)
    }


@fixture(scope='module')
def setup_scheduling_inner_params(request, setup_wg, setup_start_date):
    work_graph = setup_wg
    contractor_list = generate_resources_pool(DefaultContractorCapacity)

    return work_graph, contractor_list, setup_start_date


@fixture(scope='function')
def setup_schedule(request, setup_scheduling_inner_params):
    work_graph, contractors, start = setup_scheduling_inner_params
    scheduler_type = hasattr(request, 'param') and request.param or SchedulerType.Topological
    return generate_schedule(scheduling_algorithm_type=scheduler_type,
                             work_time_estimator=None,
                             work_graph=work_graph,
                             contractors=contractors,
                             validate_schedule=False), scheduler_type
