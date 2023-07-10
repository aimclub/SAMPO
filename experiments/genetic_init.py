from uuid import uuid4

from sampo.generator import SimpleSynthetic
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker


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


GRAPH_SIZE = 1000
BORDER_RADIUS = 20

ss = SimpleSynthetic()
wg = ss.work_graph(top_border=GRAPH_SIZE)
contractors = generate_contractors(wg, 1, 0)


baseline_genetic = GeneticScheduler(mutate_order=1.0,
                                    mutate_resources=1.0,
                                    size_selection=200,
                                    size_of_population=200)

optimized_genetic = GeneticScheduler(mutate_order=1.0,
                                     mutate_resources=1.0,
                                     size_selection=200,
                                     size_of_population=200)
optimized_genetic.set_weights([14, 11, 1, 1, 1, 1, 10])

baseline_result = baseline_genetic.schedule(wg, contractors)
my_result = optimized_genetic.schedule(wg, contractors)

print(f'Graph size: {wg.vertex_count}')
print(f'Baseline result: {baseline_result.execution_time}')
print(f'My result: {my_result.execution_time}')
