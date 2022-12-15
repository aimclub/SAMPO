from random import Random

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import generate_queues
from sampo.scheduler.multi_agency.block_validation import validate_block_schedule
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.obstruction import OneInsertObstruction

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return OneInsertObstruction.from_static_graph(1, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))


def run_tests(schedulers: list[Scheduler]):
    print('---- Variation: contractor size ----')
    for i in range(1, 5):
        print(f'--| Iteration {i}, contractor_size = {10 * i}')

        contractors = [p_rand.contactor(10 * i) for _ in range(len(schedulers))]

        agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                  for i, contractor in enumerate(contractors)]
        manager = Manager(agents)

        bg = generate_queues([1, 1, 1], lambda x: (100, 200), rand, obstruction_getter,
                             10,
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9],
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9])

        scheduled_blocks = manager.manage_blocks(bg, log=True)
        validate_block_schedule(bg, scheduled_blocks)

    print('---- Variation: block size ----')
    for i in range(1, 5):
        print(f'--| Iteration {i}, block_size ~ {50 * i}')

        contractors = [p_rand.contactor(10) for _ in range(len(schedulers))]

        agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                  for i, contractor in enumerate(contractors)]
        manager = Manager(agents)

        bg = generate_queues([1, 1, 1], lambda x: (50 * i, 50 * (i + 1)), rand, obstruction_getter,
                             10,
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9],
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9])

        scheduled_blocks = manager.manage_blocks(bg, log=True)
        validate_block_schedule(bg, scheduled_blocks)


print('------ Genetics tests begin ------')

schedulers = [GeneticScheduler(5, 50, 0.5, 0.5, 50),
              GeneticScheduler(5, 100, 0.5, 0.5, 50),
              GeneticScheduler(5, 100, 0.5, 0.5, 100),
              GeneticScheduler(5, 100, 0.75, 0.75, 50),
              GeneticScheduler(5, 50, 0.9, 0.9, 50)]
run_tests(schedulers)

print('------ All algorithms tests begin ------')
schedulers = [HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler(), GeneticScheduler()]
run_tests(schedulers)
