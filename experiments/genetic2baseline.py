from uuid import uuid4

from sampo.generator import SimpleSynthetic
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker


class BaselineGeneticScheduler(GeneticScheduler):
    def generate_first_population(self, wg: WorkGraph, contractors: list[Contractor]):
        return {
            "heft_end": (None, None),
            "heft_between": (None, None),
            "12.5%": (None, None),
            "25%": (None, None),
            "75%": (None, None),
            "87.5%": (None, None)
        }


def generate_contractors(wg: WorkGraph, num_contractors: int, contractor_min_resources: int) -> list[Contractor]:
    resource_req: dict[str, int] = {}
    resource_req_count: dict[str, int] = {}

    for node in wg.nodes:
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


GRAPH_SIZE = 100
BORDER_RADIUS = 20

ss = SimpleSynthetic(231)
wg = ss.work_graph(bottom_border=GRAPH_SIZE - BORDER_RADIUS,
                   top_border=GRAPH_SIZE + BORDER_RADIUS)
contractors = generate_contractors(wg, 1, 0)

baseline_result = BaselineGeneticScheduler().schedule(wg, contractors)
my_result = GeneticScheduler().schedule(wg, contractors)
print(f'Baseline result: {baseline_result.execution_time}')
print(f'My result: {my_result.execution_time}')
