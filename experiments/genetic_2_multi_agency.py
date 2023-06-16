from random import Random

from sampo.generator import SimpleSynthetic
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.multi_agency.block_generator import SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.schemas.time import Time

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return None


if __name__ == '__main__':
    schedulers = [GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  GeneticScheduler(50, 50, 0.5, 0.5, 100)]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    bg = generate_block_graph(SyntheticBlockGraphType.Queues, 100, [1, 1, 1], lambda x: (30, 50), 1.0,
                              rand, obstruction_getter, 2, [3, 4], [3, 4], logger=print)

    scheduled_blocks = manager.manage_blocks(bg, logger=print)

    min_downtime = Time.inf()
    best_genetic = None
    for agent in agents:
        if agent.downtime < min_downtime:
            min_downtime = agent.downtime
            best_genetic = agent.scheduler

    conjuncted = bg.to_work_graph()

    print(f'Best genetic: {best_genetic}')
    print(f'Conjunction finished: {conjuncted.vertex_count} works')

    schedule = best_genetic.schedule(conjuncted, contractors)

    ma_res = max(sblock.end_time for sblock in scheduled_blocks.values())
    genetic_res = schedule.execution_time

    with open(f'ma2genetic_results_{Random().randint(0, 10000000)}.txt', 'w') as f:
        f.write(f'Conjunction finished: {conjuncted.vertex_count} works\n')
        f.write(f'Multi-agency res: {ma_res}\n')
        f.write(f'Genetic res: {genetic_res}\n')
        f.write(f'MA / Genetic win is {(1 - ma_res / genetic_res) * 100}%\n')
