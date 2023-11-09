import time

from pathos.multiprocessing import ProcessingPool

from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.schemas.contractor import DEFAULT_CONTRACTOR_CAPACITY


def run_iteration(args) -> int:
    algo_ind, graph_size = args
    ss = SimpleSynthetic(rand=231)
    wg = ss.work_graph(SyntheticGraphType.PARALLEL, graph_size - 50, graph_size + 50)
    contractors = [ss.contractor(DEFAULT_CONTRACTOR_CAPACITY)]

    scheduler_type = list(SchedulerType)[algo_ind]

    start = time.time()
    generate_schedule(scheduler_type, None, wg, contractors, validate_schedule=False)
    return int(time.time() - start)


if __name__ == '__main__':
    with ProcessingPool(nodes=4) as p:
        graph_sizes = [250, 500, 1000, 5000, 10000]
        args = [(algo_ind, graph_size) for algo_ind in range(len(SchedulerType)) for graph_size in graph_sizes]
        measurements = p.map(run_iteration, args)

        with open('algorithms_performance.txt', 'w') as f:
            f.write(' '.join([str(val) for val in measurements]))
