from random import Random

from sampo.generator import SimpleSynthetic
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.multi_agency.block_generator import SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.schemas.time import Time

r_seed = 231 + Random(0).randint(0, 1000000)
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)

print(f'Seed: {r_seed}')


def obstruction_getter(i: int):
    return None


if __name__ == '__main__':
    schedulers = [GeneticScheduler(50, 50, 0.5, 0.5, 100),]
                  # GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  # GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  # GeneticScheduler(50, 50, 0.5, 0.5, 100),
                  # GeneticScheduler(50, 50, 0.5, 0.5, 100)]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    bg = generate_block_graph(SyntheticBlockGraphType.SEQUENTIAL, 100, [1, 0, 0], lambda x: (None, 50), 0.2,
                              rand, obstruction_getter, 2, [3, 4] * 1, [3, 4] * 1, logger=print)

    conjuncted = bg.to_work_graph()
    print(f'Conjunction finished: {conjuncted.vertex_count} works')

    # schedule = agents[0].scheduler.schedule(conjuncted, contractors)

    scheduled_blocks = manager.manage_blocks(bg, logger=print)

    min_downtime = Time.inf()
    best_genetic = None
    for agent in agents:
        if agent.downtime < min_downtime:
            min_downtime = agent.downtime
            best_genetic = agent.scheduler

    print(f'Best genetic: {best_genetic}')

    schedule = best_genetic.schedule(conjuncted, contractors)

    ma_res = max(sblock.end_time for sblock in scheduled_blocks.values())
    genetic_res = schedule.execution_time

    filename = f'ma2genetic_results_{Random().randint(0, 10000000)}.txt'
    with open(filename, 'w') as f:
        f.write(f'Conjunction finished: {conjuncted.vertex_count} works\n')
        f.write(f'Multi-agency res: {ma_res}\n')
        f.write(f'Genetic res: {genetic_res}\n')
        f.write(f'MA / Genetic win is {int((1 - ma_res / genetic_res) * 100)}%\n')

    print(f'Results saved to {filename}')
