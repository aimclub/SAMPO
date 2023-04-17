from random import Random
from typing import Iterable

from sampo.generator import SimpleSynthetic
from sampo.generator.pipeline.types import SyntheticGraphType
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.multi_agency.block_generator import generate_blocks, SyntheticBlockGraphType, generate_queues
from sampo.scheduler.multi_agency.block_validation import validate_block_schedule
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager, ScheduledBlock
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.obstruction import OneInsertObstruction
from sampo.schemas.contractor import Contractor, DefaultContractorCapacity


def test_one_auction():
    p_rand = SimpleSynthetic(rand=231)
    contractors = [p_rand.contractor(i) for i in range(10, 101, 10)]

    agents = [Agent(f'Agent {i}', HEFTScheduler(), [contractor]) for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    for i in range(10):
        p_rand = SimpleSynthetic(rand=231 + i)
        wg = p_rand.work_graph(top_border=200)
        start_time, end_time, schedule, agent = manager.run_auction(wg)

        print(f'Round {i}: wins {agent} with submitted time {end_time - start_time}')


def manage_block_graph(contractors: list[Contractor]):
    r_seed = 231
    rand = Random(r_seed)

    scheduler_constructors = [HEFTScheduler, HEFTBetweenScheduler, TopologicalScheduler, GeneticScheduler]

    agents = [Agent(f'Agent {i}', scheduler_constructors[i % len(scheduler_constructors)](), [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)
    bg = generate_blocks(SyntheticBlockGraphType.Random, 10, [1, 1, 1], lambda x: (100, 200), 0.5, rand)

    scheduled_blocks = manager.manage_blocks(bg, logger=print)

    validate_block_schedule(bg, scheduled_blocks, agents)

    for sblock in scheduled_blocks.values():
        print(sblock)


def test_manage_block_graph_different_contractors():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    contractors = [p_rand.contractor(i) for i in range(10, 101, 10)]
    manage_block_graph(contractors)


def test_manage_block_graph_same_contractors():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    contractors = [p_rand.contractor(DefaultContractorCapacity) for _ in range(4)]
    manage_block_graph(contractors)


def test_managing_with_obstruction():
    r_seed = 231
    p_rand = SimpleSynthetic(rand=r_seed)
    rand = Random(r_seed)
    contractors = [p_rand.contractor(10)]

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

        scheduled_blocks = manager.manage_blocks(bg, logger=print)
        validate_block_schedule(bg, scheduled_blocks, agents)

        scheduled_blocks_without_obstructions = manager.manage_blocks(bg_without_obstructions, logger=print)
        validate_block_schedule(bg_without_obstructions, scheduled_blocks_without_obstructions, agents)

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

    def obstruction_getter(i: int):
        return OneInsertObstruction.from_static_graph(1, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))

    for i in range(5):
        contractors = [p_rand.contractor(10)]

        scheduler_constructors = [HEFTScheduler]

        agents = [Agent(f'Agent {i}', scheduler_constructors[i % len(scheduler_constructors)](), [contractor])
                  for i, contractor in enumerate(contractors)]
        manager = Manager(agents)

        bg = generate_queues([1, 1, 1], lambda x: (100, 200), rand, obstruction_getter,
                             10,
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9],
                             [3, 4, 6, 8, 10, 3, 4, 8, 9, 9])

        scheduled_blocks = manager.manage_blocks(bg, logger=print)
        validate_block_schedule(bg, scheduled_blocks, agents)

        for sblock in scheduled_blocks.values():
            print(sblock)
