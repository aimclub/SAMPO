from functools import partial
from random import Random
from typing import IO

from sampo.generator import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.time import Time

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return None


def log(message: str, logfile: IO):
    # print(message)
    logfile.write(message + '\n')


if __name__ == '__main__':
    schedulers = [HEFTScheduler(),
                  HEFTBetweenScheduler(),
                  TopologicalScheduler()]
                  # GeneticScheduler(50, 50, 0.5, 0.5, 20)]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    with open(f'algorithms_2_multi_agency_comparison.txt', 'w') as logfile:
        logger = partial(log, logfile=logfile)

        bg = generate_block_graph(SyntheticBlockGraphType.RANDOM, 10, [0, 1, 1], lambda x: (None, 50), 0.5,
                                  rand, obstruction_getter, 2, [3, 4], [3, 4], logger=logger)
        conjuncted = bg.to_work_graph()

        scheduled_blocks = manager.manage_blocks(bg, logger=logger)

        print(f'Multi-agency res: {max(sblock.end_time for sblock in scheduled_blocks.values())}')

        min_downtime = Time.inf()
        best_algo = None
        for agent in agents:
            if agent.downtime < min_downtime:
                min_downtime = agent.downtime
                best_algo = agent.scheduler

        print(f'Best algo: {best_algo}')

        schedule = best_algo.schedule(conjuncted, contractors)

        print(f'Best algo res: {schedule.execution_time}')
