from pytest import fixture

from sampo.generator.environment.contractor_by_wg import ContractorGenerationMethod, get_contractor_by_wg
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph


@fixture
def setup_empty_req_work_graph(setup_simple_synthetic) -> WorkGraph:
    wg = setup_simple_synthetic.work_graph(top_border=100)

    for node in wg.nodes:
        node.work_unit.worker_reqs.clear()
    return wg


@fixture
def setup_empty_contractors(setup_empty_req_work_graph) -> list[Contractor]:
    return [get_contractor_by_wg(setup_empty_req_work_graph, method=ContractorGenerationMethod.MIN)]


def test_empty_graph_empty_contractor(setup_empty_req_work_graph, setup_empty_contractors, setup_scheduler):
    scheduler, scheduler_type = setup_scheduler
    schedule = scheduler.schedule(setup_empty_req_work_graph,
                                  setup_empty_contractors,
                                  validate=False)

    assert not schedule.execution_time.is_inf()
