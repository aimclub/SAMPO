from pathos.multiprocessing import ProcessingPool

from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.time import Time


def run_test(args) -> list[tuple[Time, Time]]:
    graph_size, iterations = args

    result = []
    for i in range(iterations):
        ss = SimpleSynthetic()
        wg = ss.work_graph(top_border=graph_size)
        contractors = [get_contractor_by_wg(wg)]

        baseline_genetic = GeneticScheduler(mutate_order=1.0,
                                            mutate_resources=1.0,
                                            size_selection=200,
                                            size_of_population=200)

        optimized_genetic = GeneticScheduler(mutate_order=1.0,
                                             mutate_resources=1.0,
                                             size_selection=200,
                                             size_of_population=200)
        optimized_genetic.set_weights([14, 11, 1, 1, 1, 1, 10])

        baseline_result = baseline_genetic.schedule(wg, contractors)
        my_result = optimized_genetic.schedule(wg, contractors)

        result.append((baseline_result.execution_time, my_result.execution_time))

    return result


if __name__ == '__main__':
    num_iterations = 4

    sizes = [100 * i for i in range(1, num_iterations + 1)]
    iterations = [5 - i for i in range(1, num_iterations + 1)]

    with ProcessingPool(10) as p:
        results_by_size = p.map(run_test, zip(sizes, iterations))

        print()
        with open('genetic_init_res.txt', 'w') as f:
            for graph_size, result_list in zip(sizes, results_by_size):
                global_ratio = 0
                for baseline_result, my_result in result_list:
                    ratio = baseline_result / my_result
                    global_ratio += ratio
                global_ratio /= len(result_list)

                res_string = f'Size: {graph_size}, upgrade ratio: {global_ratio}'
                print(res_string)
                f.write(res_string)
                f.write('\n')
