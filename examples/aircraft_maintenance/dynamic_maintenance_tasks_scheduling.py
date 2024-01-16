from random import Random
from typing import Dict

from sampo.scheduler.base import Scheduler
from sampo.scheduler.multi_agency import validate_block_schedule
from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager, ScheduledBlock
from sampo.scheduler.utils.obstruction import OneInsertObstruction, Obstruction
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph


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


def run_example(queues_with_obstructions: list[list[WorkGraph]],
                schedulers: list[Scheduler], contractors: list[Contractor]) -> Dict[str, ScheduledBlock]:

    # Scheduling agents and manager initialization
    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    # Upload information about routine tasks, obstruction tasks, related to them and probabilities
    bg = load_queues_bg(queues_with_obstructions)

    # Schedule blocks of tasks using multi-agent modelling
    blocks_schedules = manager.manage_blocks(bg, logger=print)

    validate_block_schedule(bg, blocks_schedules, agents)

    return blocks_schedules
