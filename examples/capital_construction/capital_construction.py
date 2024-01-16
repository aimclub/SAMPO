from typing import Dict

from sampo.scheduler.base import Scheduler
from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.scheduler.multi_agency.multi_agency import Agent, ScheduledBlock, StochasticManager
from sampo.schemas import IntervalGaussian
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time_estimator import DefaultWorkEstimator


def load_queues_bg(queues: list[list[WorkGraph]]):
    wgs: list[WorkGraph] = [wg for queue in queues for wg in queue]

    bg = BlockGraph.pure(wgs)

    index = 0  # global wg index in `wgs`
    nodes_prev = []
    for queue in queues:
        nodes = [bg[wgs[i].start.id] for i in range(index, index + len(queue))]

        # generate edges
        generated_edges = 0
        for i, node in enumerate(nodes[:-2]):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[i])
            generated_edges += 1

        for i, node in enumerate(nodes):
            if i >= len(nodes_prev):
                break
            # we are going in reverse to fill edges that are not covered by previous cycle
            BlockGraph.add_edge(node, nodes_prev[-i])
            generated_edges += 1

        nodes_prev = nodes

    return bg


def run_example(queues: list[list[WorkGraph]],
                schedulers: list[Scheduler],
                contractors: list[Contractor]) -> Dict[str, ScheduledBlock]:
    work_estimator = DefaultWorkEstimator()
    for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']:
        for i, contractor in enumerate(contractors):
            work_estimator.set_worker_productivity(IntervalGaussian(0.2 * i + 0.2, 1, 0, 2), worker, contractor.id)
    for scheduler in schedulers:
        scheduler.work_estimator = work_estimator

    # Scheduling agents and manager initialization
    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = StochasticManager(agents)

    # Upload information about routine tasks
    bg = load_queues_bg(queues)

    # Schedule blocks of tasks using multi-agent modelling
    blocks_schedules = manager.manage_blocks(bg, logger=print)

    return blocks_schedules
