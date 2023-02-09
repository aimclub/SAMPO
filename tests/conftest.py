from collections import defaultdict
from random import Random
from typing import Dict, List, Optional
from uuid import uuid4

import pytest
from pytest import fixture

from sampo.generator import SimpleSynthetic
from sampo.generator.pipeline.project import get_start_stage, get_finish_stage
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, EdgeType
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.structurator.base import graph_restructuring
from sampo.utilities.sampler import Sampler

pytest_plugins = ("tests.schema", "tests.models",)


@fixture(scope='session')
def setup_sampler(request):
    return Sampler(1e-1)


@fixture(scope='session')
def setup_rand() -> Random:
    return Random(231)


@fixture(scope='session')
def setup_simple_synthetic(setup_rand) -> SimpleSynthetic:
    return SimpleSynthetic(setup_rand)


@fixture(scope='session',
         params=[(graph_type, lag) for lag in [True, False]
                 for graph_type in ['manual',
                                    'small plain synthetic', 'big plain synthetic',
                                    'small advanced synthetic', 'big advanced synthetic']],
         ids=[f'Graph: {graph_type}, LAG_OPT={lag_opt}'
              for lag_opt in [True, False]
              for graph_type in ['manual',
                                 'small plain synthetic', 'big plain synthetic',
                                 'small advanced synthetic', 'big advanced synthetic']])
def setup_wg(request, setup_sampler, setup_simple_synthetic) -> WorkGraph:
    SMALL_GRAPH_SIZE = 100
    BIG_GRAPH_SIZE = 1000
    BORDER_RADIUS = 20
    ADV_GRAPH_UNIQ_WORKS_PROP = 0.4
    ADV_GRAPH_UNIQ_RES_PROP = 0.2

    graph_type, lag_optimization = request.param

    match graph_type:
        case 'manual':
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
        case 'small plain synthetic':
            wg = setup_simple_synthetic.work_graph(bottom_border=SMALL_GRAPH_SIZE - BORDER_RADIUS,
                                                   top_border=SMALL_GRAPH_SIZE + BORDER_RADIUS)
        case 'big plain synthetic':
            wg = setup_simple_synthetic.work_graph(bottom_border=BIG_GRAPH_SIZE - BORDER_RADIUS,
                                                   top_border=BIG_GRAPH_SIZE + BORDER_RADIUS)
        case 'small advanced synthetic':
            size = SMALL_GRAPH_SIZE + BORDER_RADIUS
            wg = setup_simple_synthetic \
                .advanced_work_graph(works_count_top_border=size,
                                     uniq_works=int(size * ADV_GRAPH_UNIQ_WORKS_PROP),
                                     uniq_resources=int(size * ADV_GRAPH_UNIQ_RES_PROP))
        case 'big advanced synthetic':
            size = BIG_GRAPH_SIZE + BORDER_RADIUS
            wg = setup_simple_synthetic \
                .advanced_work_graph(works_count_top_border=size,
                                     uniq_works=int(size * ADV_GRAPH_UNIQ_WORKS_PROP),
                                     uniq_resources=int(size * ADV_GRAPH_UNIQ_RES_PROP))
        case _:
            raise ValueError(f'Unknown graph type: {graph_type}')

    if lag_optimization:
        wg = graph_restructuring(wg, use_lag_edge_optimization=True)

    return wg


@fixture(scope='session')
def setup_worker_pool(setup_contractors) -> WorkerContractorPool:
    worker_pool = defaultdict(dict)
    for contractor in setup_contractors:
        for worker in contractor.workers.values():
            worker_pool[worker.name][worker.contractor_id] = worker
    return worker_pool


# TODO Make parametrization with different(specialized) contractors
@fixture(scope='session',
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
                                      workers={name: Worker(str(uuid4()), name, count, contractor_id=contractor_id)
                                               for name, count in resource_req.items()},
                                      equipments={}))
    return contractors


@fixture(scope='session')
def setup_default_schedules(setup_wg, setup_contractors):
    work_estimator: Optional[WorkTimeEstimator] = None

    def init_schedule(scheduler_class):
        return scheduler_class(work_estimator=work_estimator).schedule(setup_wg, setup_contractors)

    try:
        return {
            "heft_end": init_schedule(HEFTScheduler),
            "heft_between": init_schedule(HEFTBetweenScheduler)
        }
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')


@fixture(scope='session',
         params=list(SchedulerType),
         ids=[f'Scheduler: {scheduler}' for scheduler in list(SchedulerType)])
def setup_scheduler_type(request):
    return request.param


@fixture(scope='session')
def setup_schedule(request, setup_wg, setup_contractors):
    scheduler_type = hasattr(request, 'param') and request.param or SchedulerType.Topological
    try:
        return generate_schedule(scheduling_algorithm_type=scheduler_type,
                                 work_time_estimator=None,
                                 work_graph=setup_wg,
                                 contractors=setup_contractors,
                                 validate_schedule=False), scheduler_type
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')
