from functools import partial
from random import Random
from typing import IO

from pathos.multiprocessing import ProcessingPool

from sampo.generator import SimpleSynthetic
from sampo.generator.pipeline.types import SyntheticGraphType
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.multi_agency.block_generator import SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.obstruction import OneInsertObstruction

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return OneInsertObstruction.from_static_graph(0.5, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))


def log(message: str, logfile: IO):
    # print(message)
    logfile.write(message + '\n')


def variate_contractor_size(logfile: IO, schedulers: list[Scheduler]):
    logger = partial(log, logfile=logfile)
    for i in range(1, 5):
        logger(f'contractor_size = {10 * i}')

        for graph_type in SyntheticBlockGraphType:
            contractors = [p_rand.contractor(10 * i) for _ in range(len(schedulers))]

            agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                      for i, contractor in enumerate(contractors)]
            manager = Manager(agents)

            bg = generate_block_graph(graph_type, 10, [1, 1, 1], lambda x: (100, 200), 0.5,
                                      rand, obstruction_getter, 2, [3, 4], [3, 4], logger=logger)

            scheduled_blocks = manager.manage_blocks(bg, logger=logger)
            # validate_block_schedule(bg, scheduled_blocks)


def variate_block_size(logfile: IO, schedulers: list[Scheduler]):
    logger = partial(log, logfile=logfile)
    for i in range(1, 5):
        logger(f'block_size ~ {50 * i}')

        for graph_type in SyntheticBlockGraphType:
            contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

            agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                      for i, contractor in enumerate(contractors)]
            manager = Manager(agents)

            bg = generate_block_graph(graph_type, 10, [1, 1, 1], lambda x: (50 * i, 50 * (i + 1)), 0.5,
                                      rand, obstruction_getter, 2, [3, 4], [3, 4], logger=logger)

            scheduled_blocks = manager.manage_blocks(bg, logger=logger)
            # validate_block_schedule(bg, scheduled_blocks)
            # downtimes
            logger(' '.join([str(agent.downtime.value) for agent in agents]))

        logger('')


def run_iteration(args):
    i, mode = args
    with open(f'algorithms_comparison_block_size_{mode}_{i}.txt', 'w') as logfile:
        if mode == 0:
            schedulers = [HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler(), GeneticScheduler()]
        else:
            schedulers = [GeneticScheduler(5, 50, 0.5, 0.5, 50),
                          GeneticScheduler(5, 100, 0.5, 0.5, 50),
                          GeneticScheduler(5, 100, 0.5, 0.75, 50),
                          GeneticScheduler(5, 100, 0.75, 0.75, 50),
                          GeneticScheduler(5, 50, 0.9, 0.9, 50)]
        variate_block_size(logfile, schedulers)


if __name__ == '__main__':
    pool = ProcessingPool()
    args = [[i, mode] for mode in [0, 1] for i in range(1)]

    pool.map(run_iteration, args)
