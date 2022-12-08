from random import Random

from sampo.generator import SimpleSynthetic
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.block_graph import generate_blocks
from sampo.scheduler.utils.block_validation import validate_block_schedule
from sampo.scheduler.utils.multi_agency import Agent, Manager


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
    bg = generate_blocks(10, [1, 1, 1], lambda x: (100, 200), 0.5, rand)

    scheduled_blocks = manager.manage_blocks(bg, log=True)

    validate_block_schedule(bg, scheduled_blocks)

    for sblock in scheduled_blocks.values():
        print(sblock)


