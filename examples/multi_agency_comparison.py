from random import Random

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import generate_queues, SyntheticBlockGraphType, generate_block_graph
from sampo.scheduler.multi_agency.block_validation import validate_block_schedule
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.utils.obstruction import OneInsertObstruction

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return OneInsertObstruction.from_static_graph(1, rand, p_rand.work_graph(SyntheticGraphType.Sequential, 10))


with open('multi_agency_comparison.txt', 'w') as logfile:
    def log(message: str):
        print(message)
        logfile.write(message + '\n')


    def run_tests(schedulers: list[Scheduler]):
        log('------ Variation: contractor size ------')
        for i in range(1, 5):
            log(f'--| Iteration {i}, contractor_size = {10 * i}\n')

            for graph_type in SyntheticBlockGraphType:
                log(f'----| Running at BlockGraph type: {graph_type}\n')
                contractors = [p_rand.contactor(10 * i) for _ in range(len(schedulers))]

                agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                          for i, contractor in enumerate(contractors)]
                manager = Manager(agents)

                bg = generate_block_graph(graph_type, 10, [1, 1, 1], lambda x: (100, 200), 0.5,
                                          rand, obstruction_getter, 2, [3, 4], [3, 4], logger=log)

                scheduled_blocks = manager.manage_blocks(bg, logger=log)
                validate_block_schedule(bg, scheduled_blocks)

        log('------ Variation: block size ------')
        for i in range(1, 5):
            log(f'--| Iteration {i}, block_size ~ {50 * i}\n')

            for graph_type in SyntheticBlockGraphType:
                log(f'----| Running at BlockGraph type: {graph_type}\n')
                contractors = [p_rand.contactor(10) for _ in range(len(schedulers))]

                agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
                          for i, contractor in enumerate(contractors)]
                manager = Manager(agents)

                bg = generate_block_graph(graph_type, 10, [1, 1, 1], lambda x: (50 * i, 50 * (i + 1)), 0.5,
                                          rand, obstruction_getter, 2, [3, 4], [3, 4], logger=log)

                scheduled_blocks = manager.manage_blocks(bg, logger=log)
                validate_block_schedule(bg, scheduled_blocks)


    # log('------------ Genetics tests begin ------------\n')
    #
    # schedulers = [GeneticScheduler(5, 50, 0.5, 0.5, 50),
    #               GeneticScheduler(5, 100, 0.5, 0.5, 50),
    #               GeneticScheduler(5, 100, 0.5, 0.5, 100),
    #               GeneticScheduler(5, 100, 0.75, 0.75, 50),
    #               GeneticScheduler(5, 50, 0.9, 0.9, 50)]
    # run_tests(schedulers)

    log('------------ All algorithms tests begin ------------\n')
    schedulers = [HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler(), GeneticScheduler()]
    run_tests(schedulers)
