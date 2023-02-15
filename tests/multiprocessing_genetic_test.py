import time

from sampo.scheduler.genetic.base import GeneticScheduler


def test_genetic_run(setup_wg, setup_contractors):
    genetic = GeneticScheduler()

    start_time = time.time()
    genetic.schedule(setup_wg, setup_contractors)
    single_core_time = time.time() - start_time

    genetic.set_use_multiprocessing(n_cpu=4)

    start_time = time.time()
    genetic.schedule(setup_wg, setup_contractors)
    multi_core_time = time.time() - start_time

    print(f'Multicore genetic ratio: {multi_core_time / single_core_time} times')
