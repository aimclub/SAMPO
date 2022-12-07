from random import Random

from sampo.generator import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.utils.block_graph import generate_blocks
from sampo.scheduler.utils.multi_agency import Agent, Manager


def test_one_auction():
    p_rand = SimpleSynthetic(rand=231)
    contractors = [p_rand.contactor(i) for i in range(10, 101, 10)]

    agents = [Agent(f'Agent {i}', HEFTScheduler(), [contractor]) for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    for i in range(10):
        p_rand = SimpleSynthetic(rand=231 + i)
        wg = p_rand.work_graph(top_border=200)
        start_time, end_time, agent = manager.run_auction(wg)

        print(f'Round {i}: wins {agent.name} with submitted time {end_time - start_time}')


def test_managing_block_graph():
    p_rand = SimpleSynthetic(rand=231)
    contractors = [p_rand.contactor(i) for i in range(10, 101, 10)]

    agents = [Agent(f'Agent {i}', HEFTScheduler(), [contractor]) for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    rand = Random(231)
    bg = generate_blocks(5, [1, 1, 1], lambda x: (100, 200), 0.5, rand)

    scheduled_blocks = manager.manage_blocks(bg)

    print(scheduled_blocks)


