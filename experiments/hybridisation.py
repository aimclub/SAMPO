import json

import pathos
from tqdm import tqdm

import sampo.scheduler
from sampo.backend.multiproc import MultiprocessingComputationalBackend

from sampo.hybrid.population_tabu import TabuPopulationScheduler

from sampo.hybrid.cycle import CycleHybridScheduler
from sampo.api.genetic_api import ScheduleGenerationScheme
from sampo.scheduler import HEFTScheduler, HEFTBetweenScheduler, TopologicalScheduler, GeneticScheduler
from sampo.hybrid.population import HeuristicPopulationScheduler, GeneticPopulationScheduler

from sampo.generator.environment import get_contractor_by_wg
from sampo.generator import SimpleSynthetic

from sampo.base import SAMPO
from sampo.schemas import WorkGraph

def run_experiment(args):
    graph_size, iteration = args

    heuristics = HeuristicPopulationScheduler([HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler()])
    # genetic1 = TabuPopulationScheduler()
    genetic1 = GeneticPopulationScheduler(GeneticScheduler(mutate_order=0.2,
                                                           mutate_resources=0.2,
                                                           sgs_type=ScheduleGenerationScheme.Parallel))
    genetic2 = GeneticPopulationScheduler(GeneticScheduler(mutate_order=0.001,
                                                           mutate_resources=0.001,
                                                           sgs_type=ScheduleGenerationScheme.Parallel))

    hybrid_combine = CycleHybridScheduler(heuristics, [genetic1, genetic2], max_plateau_size=1)
    hybrid_genetic1 = CycleHybridScheduler(heuristics, [genetic1], max_plateau_size=1)
    hybrid_genetic2 = CycleHybridScheduler(heuristics, [genetic2], max_plateau_size=1)

    wg = WorkGraph.load('wgs', f'{graph_size}_{iteration}')
    contractors = [get_contractor_by_wg(wg)]

    # SAMPO.backend = MultiprocessingComputationalBackend(n_cpus=10)
    SAMPO.backend.cache_scheduler_info(wg, contractors)
    SAMPO.backend.cache_genetic_info()

    schedule_hybrid_combine = hybrid_combine.schedule(wg, contractors)
    schedule_genetic1 = hybrid_genetic1.schedule(wg, contractors)
    schedule_genetic2 = hybrid_genetic2.schedule(wg, contractors)

    # print(f'Hybrid combine: {schedule_hybrid_combine.execution_time}')
    # print(f'Scheduler 1 cycled: {schedule_genetic1.execution_time}')
    # print(f'Scheduler 2 cycled: {schedule_genetic2.execution_time}')
    return schedule_hybrid_combine.execution_time, schedule_genetic1.execution_time, schedule_genetic2.execution_time

if __name__ == '__main__':
    arguments = [(graph_size, iteration) for graph_size in [100, 200, 300, 400, 500] for iteration in range(5)]
    results = {graph_size: [] for graph_size in [100, 200, 300, 400, 500]}

    with pathos.multiprocessing.Pool(processes=11) as p:
        r = p.map(run_experiment, arguments)

        for (graph_size, _), (combined_time, time1, time2) in zip(arguments, r):
            results[graph_size].append((combined_time / time1, combined_time / time2))

    with open('hybrid_results.json', 'w') as f:
        json.dump(results, f)
