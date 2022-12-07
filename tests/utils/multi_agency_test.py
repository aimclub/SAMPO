from sampo.generator import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.utils.multi_agency import Agent, Manager


def test_one_auction():
    p_rand = SimpleSynthetic(rand=231)
    contractors = [p_rand.contactor(i) for i in range(10, 101, 10)]

    agents = [Agent(f'Agent {i}', HEFTScheduler(), [contractor]) for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    for i in range(10):
        p_rand = SimpleSynthetic(rand=231 + i)
        wg = p_rand.work_graph(top_border=200)
        time, agent = manager.run_auction(wg)

        print(f'Round {i}: wins {agent.name} with submitted time {time}')


