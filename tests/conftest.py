from random import Random
from typing import Dict, Any
from uuid import uuid4

import pytest
from pytest import fixture

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler import HEFTScheduler, HEFTBetweenScheduler, TopologicalScheduler, SchedulerType, Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, EdgeType
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.landscape import LandscapeConfiguration, ResourceHolder
from sampo.schemas.requirements import MaterialReq
from sampo.schemas.resources import Material
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.structurator.base import graph_restructuring
from sampo.utilities.sampler import Sampler

pytest_plugins = ('tests.schema', 'tests.models',)


@fixture
def setup_sampler(request):
    return Sampler(1e-1)


@fixture
def setup_rand() -> Random:
    return Random(231)


@fixture
def setup_landscape_one_holder():
    return LandscapeConfiguration(holders=[ResourceHolder(str(uuid4()), 'holder1', IntervalGaussian(25, 0),
                                                          materials=[Material('111', 'mat1', 100000)])])


@fixture
def setup_landscape_many_holders():
    return LandscapeConfiguration(holders=[ResourceHolder(str(uuid4()), 'holder1', IntervalGaussian(50, 0),
                                                          materials=[Material('111', 'mat1', 100000)]),
                                           ResourceHolder(str(uuid4()), 'holder2', IntervalGaussian(50, 0),
                                                          materials=[Material('222', 'mat2', 100000)])
                                           ])


@fixture
def setup_simple_synthetic(setup_rand) -> SimpleSynthetic:
    return SimpleSynthetic(setup_rand)


@fixture(params=[(graph_type, lag) for lag in [True, False]
                 for graph_type in ['manual',
                                    'small plain synthetic', 'big plain synthetic']],
         # 'small advanced synthetic', 'big advanced synthetic']],
         ids=[f'Graph: {graph_type}, LAG_OPT={lag_opt}'
              for lag_opt in [True, False]
              for graph_type in ['manual',
                                 'small plain synthetic', 'big plain synthetic']])
# 'small advanced synthetic', 'big advanced synthetic']])
def setup_wg(request, setup_sampler, setup_simple_synthetic) -> WorkGraph:
    SMALL_GRAPH_SIZE = 100
    BIG_GRAPH_SIZE = 300
    BORDER_RADIUS = 20
    ADV_GRAPH_UNIQ_WORKS_PROP = 0.4
    ADV_GRAPH_UNIQ_RES_PROP = 0.2

    graph_type, lag_optimization = request.param

    match graph_type:
        case 'manual':
            sr = setup_sampler

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


# TODO Make parametrization with different(specialized) contractors
@fixture(params=[(i, 5 * j) for j in range(2) for i in range(1, 2)],
         ids=[f'Contractors: count={i}, min_size={5 * j}' for j in range(2) for i in range(1, 2)])
def setup_scheduler_parameters(request, setup_wg, setup_landscape_many_holders) -> tuple[
    WorkGraph, list[Contractor], LandscapeConfiguration | Any]:
    resource_req: Dict[str, int] = {}
    resource_req_count: Dict[str, int] = {}

    num_contractors, contractor_min_resources = request.param

    for node in setup_wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_req[req.kind] = max(contractor_min_resources,
                                         resource_req.get(req.kind, 0) + (req.min_count + req.max_count) // 2)
            resource_req_count[req.kind] = resource_req_count.get(req.kind, 0) + 1

    for req in resource_req.keys():
        resource_req[req] = resource_req[req] // resource_req_count[req] + 1

    for node in setup_wg.nodes:
        for req in node.work_unit.worker_reqs:
            assert resource_req[req.kind] >= req.min_count

    # contractors are the same and universal(but multiple)
    contractors = []
    for i in range(num_contractors):
        contractor_id = str(uuid4())
        contractors.append(Contractor(id=contractor_id,
                                      name='OOO Berezka',
                                      workers={name: Worker(str(uuid4()), name, count * 100, contractor_id=contractor_id)
                                               for name, count in resource_req.items()},
                                      equipments={}))
    return setup_wg, contractors, setup_landscape_many_holders


@fixture
def setup_empty_contractors(setup_wg) -> list[Contractor]:
    resource_req: set[str] = set()

    num_contractors= 1

    for node in setup_wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_req.add(req.kind)

    contractors = []
    for i in range(num_contractors):
        contractor_id = str(uuid4())
        contractors.append(Contractor(id=contractor_id,
                                      name='OOO Berezka',
                                      workers={name: Worker(str(uuid4()), name, 0, contractor_id=contractor_id)
                                               for name in resource_req},
                                      equipments={}))
    return contractors


@fixture
def setup_default_schedules(setup_scheduler_parameters):
    work_estimator: WorkTimeEstimator = DefaultWorkEstimator()

    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    return setup_scheduler_parameters, GeneticScheduler.generate_first_population(setup_wg, setup_contractors,
                                                                                  setup_landscape,
                                                                                  work_estimator=work_estimator)


@fixture(params=list(SchedulerType),
         ids=[f'Scheduler: {scheduler.value}' for scheduler in list(SchedulerType)])
def setup_scheduler_type(request):
    return request.param


@fixture(params=[HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler(), GeneticScheduler(3)],
         ids=['HEFTScheduler', 'HEFTBetweenScheduler', 'TopologicalScheduler', 'GeneticScheduler'])
def setup_scheduler(request) -> Scheduler:
    return request.param


@fixture
def setup_schedule(setup_scheduler, setup_scheduler_parameters, setup_landscape_many_holders):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters

    try:
        return setup_scheduler.schedule(setup_wg,
                                        setup_contractors,
                                        validate=False,
                                        landscape=landscape), setup_scheduler_type, setup_scheduler_parameters
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')
