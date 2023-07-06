from functools import partial
from random import Random
from typing import IO

from pathos.multiprocessing import ProcessingPool

from sampo.generator import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.multi_agency.block_generator import SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.topological.base import TopologicalScheduler

r_seed = Random().randint(0, 100000)
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return None
    # return OneInsertObstruction.from_static_graph(0.5, rand, p_rand.work_graph(SyntheticGraphType.SEQUENTIAL, 10))


def log(message: str, logfile: IO):
    # print(message)
    logfile.write(message + '\n')


def run_iteration(args):
    i = args[0]
    with open(f'algorithms_efficiency_iteration_{i}.txt', 'w') as logfile:
        schedulers = [HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler()]

        blocks_received = {str(scheduler): 0 for scheduler in schedulers}

        logger = partial(log, logfile=logfile)
        for i in range(1, 2):
            logger(f'block_size ~ {50 * i}')

            for graph_type in SyntheticBlockGraphType:
                contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

                agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                          for i, contractor in enumerate(contractors)]
                manager = Manager(agents)

                bg = generate_block_graph(graph_type, 10, [1, 1, 1], lambda x: (50 * i, 50 * (i + 1)), 0.5,
                                          rand, obstruction_getter, 2, [3, 4], [3, 4], logger=logger)

                scheduled_blocks = manager.manage_blocks(bg, logger=logger)

                # aggregate statistics
                for sblock in scheduled_blocks.values():
                    blocks_received[str(sblock.agent.scheduler)] += 1

                # downtimes
                logger(' '.join([str(agent.downtime.value) for agent in agents]))

            logger('')

        print('Received blocks statistics:')
        for scheduler, blocks in blocks_received.items():
            print(f'{scheduler} {blocks}')


if __name__ == '__main__':
    pool = ProcessingPool(10)
    args = [[i] for i in range(10)]

    pool.map(run_iteration, args)
