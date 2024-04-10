import random

import pandas as pd

from sampo.generator import SimpleSynthetic
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import DefaultInputPipeline
from sampo.scheduler import GeneticScheduler
from sampo.schemas.time_estimator import DefaultWorkEstimator

work_time_estimator = DefaultWorkEstimator()

def run_test(args):
    graph_size, iterations, seed = args

    result = []
    for i in range(iterations):
        rand = random.Random(seed)
        ss = SimpleSynthetic(rand=rand)
        if graph_size < 100:
            wg = ss.small_work_graph()
        else:
            wg = ss.work_graph(top_border=graph_size)

        wg = ss.set_materials_for_wg(wg)
        contractors = [get_contractor_by_wg(wg, contractor_id=str(i), contractor_name='Contractor' + ' ' + str(i + 1))
                       for i in range(1)]

        landscape = ss.synthetic_landscape(wg)
        scheduler = GeneticScheduler(number_of_generation=2,
                                     mutate_order=0.05,
                                     mutate_resources=0.005,
                                     size_of_population=10,
                                     work_estimator=work_time_estimator,
                                     rand=rand)
        schedule = DefaultInputPipeline() \
            .wg(wg) \
            .contractors(contractors) \
            .work_estimator(work_time_estimator) \
            .landscape(landscape) \
            .schedule(scheduler) \
            .finish()
        result.append(schedule[0].schedule.execution_time)

    return result


if __name__ == '__main__':
    # Number of iterations for each graph size
    total_iters = 1
    # Number of graph sizes
    graphs = 6
    # Graph sizes
    sizes = [100 * i for i in range(1, graphs + 1)]
    total_results = []
    # Seed for random number generator can be specified here
    seed = 1
    # Iterate over graph sizes and receive results
    for size in sizes:
        print(size)
        print(seed)
        results_by_size = run_test((size, total_iters, seed))
        seed += 1
        total_results.append(results_by_size)

    # Save results to the DataFrame
    result_df = {'size': [], 'makespan': []}
    for i, results_by_size in enumerate(total_results):
        result = results_by_size[0]

        result_df['size'].append(sizes[i])
        result_df['makespan'].append(result)

    pd.DataFrame(result_df).to_csv('landscape_genetic_results2.csv', index=False)

