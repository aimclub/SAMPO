import time

import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.schemas.contractor import DefaultContractorCapacity


def run_iteration(args) -> int:
    algo_ind, graph_size = args
    ss = SimpleSynthetic(rand=231)
    wg = ss.work_graph(SyntheticGraphType.Parallel, graph_size - 50, graph_size + 50)
    contractors = [ss.contractor(DefaultContractorCapacity)]

    scheduler_type = list(SchedulerType)[algo_ind]

    start = time.time()
    generate_schedule(scheduler_type, None, wg, contractors, validate_schedule=False)
    return int(time.time() - start)


if __name__ == '__main__':
    with ProcessingPool(nodes=4) as p:
        graph_sizes = [250, 500, 1000, 5000, 10000]
        args = [(algo_ind, graph_size) for algo_ind in range(len(SchedulerType) - 1) for graph_size in graph_sizes]
        measurements = p.map(run_iteration, args)

        with open('algorithms_performance.txt', 'w') as f:
            f.write(' '.join([str(val) for val in measurements]))

    algo_res: list[list[int]] = []

    with open('algorithms_performance.txt', 'r') as f:
        data = [int(v) for v in f.readline().split(' ')]
        for i in range(4):
            algo_res.append(data[(i * 5):((i + 1) * 5)])

    algo_labels = ['Topological', 'HEFTAddEnd', 'HEFTAddBetween']  # , 'Genetic']
    graph_sizes = ['250', '500', '1000', '5000', '10000']

    fig = plt.figure()
    for i, algo in enumerate(algo_labels):
        ax = fig.add_subplot(221 + i)
        ax.set_title(algo)
        ax.plot(graph_sizes, algo_res[i])

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.84,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()
