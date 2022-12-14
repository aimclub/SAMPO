from random import Random
from typing import Iterable

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.block_generator import generate_blocks, SyntheticBlockGraphType, generate_queues
from sampo.scheduler.utils.block_validation import validate_block_schedule
from sampo.scheduler.utils.multi_agency import Agent, Manager, ScheduledBlock
from sampo.scheduler.utils.obstruction import OneInsertObstruction


def test_one_auction():
    p_rand = SimpleSynthetic(rand=231)
    contractors = [p_rand.contactor(i) for i in range(10, 101, 10)]

    agents = [Agent(f'Agent {i}', HEFTScheduler(), [contractor]) for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    for i in range(10):
        p_rand = SimpleSynthetic(rand=231 + i)
        wg = p_rand.work_graph(top_border=200)
        start_time, end_time, schedule, agent = manager.run_auction(wg)

        print(f'Round {i}: wins {agent.name} with submitted time {end_time - start_time}')


def test_managing_block_graph():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    rand = Random(r_seed)
    contractors = [p_rand.contactor(i) for i in range(10, 101, 10)]

    scheduler_constructors = [HEFTScheduler, HEFTBetweenScheduler, TopologicalScheduler, GeneticScheduler]

    agents = [Agent(f'Agent {i}', scheduler_constructors[i % len(scheduler_constructors)](), [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)
    bg = generate_blocks(SyntheticBlockGraphType.Random, 10, [1, 1, 1], lambda x: (100, 200), 0.5, rand)

    scheduled_blocks = manager.manage_blocks(bg, log=True)

    validate_block_schedule(bg, scheduled_blocks)

    for sblock in scheduled_blocks.values():
        print(sblock)


def test_managing_with_obstruction():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    rand = Random(r_seed)
    contractors = [p_rand.contactor(10)]

    scheduler_constructors = [HEFTScheduler]

    agents = [Agent(f'Agent {i}', scheduler_constructors[i % len(scheduler_constructors)](), [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    def obstruction_getter(i: int):
        return OneInsertObstruction.from_static_graph(1, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))

    for i in range(20):
        bg_without_obstructions = \
            generate_blocks(SyntheticBlockGraphType.Random, 1, [1, 1, 1], lambda x: (100, 200), 0.5, rand)
        bg = \
            generate_blocks(SyntheticBlockGraphType.Random, 1, [1, 1, 1], lambda x: (100, 200), 0.5, rand,
                            obstruction_getter)

        scheduled_blocks = manager.manage_blocks(bg, log=True)
        validate_block_schedule(bg, scheduled_blocks)

        scheduled_blocks_without_obstructions = manager.manage_blocks(bg_without_obstructions, log=True)
        validate_block_schedule(bg_without_obstructions, scheduled_blocks_without_obstructions)

        def finish_time(schedule: Iterable[ScheduledBlock]):
            return max([sblock.end_time for sblock in schedule])

        assert finish_time(scheduled_blocks_without_obstructions.values()) \
               > finish_time(scheduled_blocks.values())

        for sblock in scheduled_blocks.values():
            print(sblock)


def test_managing_queues():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    rand = Random(r_seed)
    contractors = [p_rand.contactor(10)]

    scheduler_constructors = [HEFTScheduler]

    agents = [Agent(f'Agent {i}', scheduler_constructors[i % len(scheduler_constructors)](), [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    def obstruction_getter(i: int):
        return OneInsertObstruction.from_static_graph(1, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))

    for i in range(20):
        bg = \
            generate_queues([1, 1, 1], lambda x: (100, 200), rand, obstruction_getter,
                            10,
                            [3, 4, 6, 8, 10, 3, 4, 8, 9, 9],
                            [3, 4, 6, 8, 10, 3, 4, 8, 9, 9])

        scheduled_blocks = manager.manage_blocks(bg, log=True)
        validate_block_schedule(bg, scheduled_blocks)

        for sblock in scheduled_blocks.values():
            print(sblock)
