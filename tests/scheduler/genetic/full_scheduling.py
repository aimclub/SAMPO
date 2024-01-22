import time

from sampo.backend import DefaultComputationalBackend
from sampo.backend.multiproc import MultiprocessingComputationalBackend
from sampo.base import SAMPO
from sampo.scheduler.genetic.base import GeneticScheduler


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    SAMPO.backend = DefaultComputationalBackend()

    genetic = GeneticScheduler(number_of_generation=10,
                               mutate_order=0.05,
                               mutate_resources=0.005,
                               size_of_population=50,
                               verbose=True)

    start_default = time.time()
    genetic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
    time_default = time.time() - start_default

    n_cpus = 10
    SAMPO.backend = MultiprocessingComputationalBackend(n_cpus=n_cpus)

    start_multiproc = time.time()
    genetic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
    time_multiproc = time.time() - start_multiproc

    print('\n------------------\n')
    print(f'Graph size: {setup_wg.vertex_count}')
    print(f'Time default: {time_default} s')
    print(f'Time multiproc: {time_multiproc} s')
    print(f'CPUs used: {n_cpus}')
    print(f'Ratio: {time_default / time_multiproc}')
