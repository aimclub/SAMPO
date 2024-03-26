import time

from sampo.api.genetic_api import ScheduleGenerationScheme
from sampo.backend.default import DefaultComputationalBackend
from sampo.backend.multiproc import MultiprocessingComputationalBackend
from sampo.backend.native import NativeComputationalBackend
from sampo.base import SAMPO
from sampo.scheduler import GeneticScheduler
from sampo.scheduler.timeline import JustInTimeTimeline
from sampo.scheduler.utils import get_worker_contractor_pool


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    SAMPO.backend = DefaultComputationalBackend()

    genetic = GeneticScheduler(number_of_generation=10,
                               mutate_order=0.05,
                               mutate_resources=0.05,
                               size_of_population=50,
                               sgs_type=ScheduleGenerationScheme.Serial)

    start_default = time.time()
    genetic.schedule(setup_wg, setup_contractors, validate=True, landscape=setup_landscape,
                     timeline=JustInTimeTimeline(get_worker_contractor_pool(setup_contractors), setup_landscape))
    time_default = time.time() - start_default

    n_cpus = 10
    SAMPO.backend = NativeComputationalBackend()

    start_multiproc = time.time()
    genetic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
    time_multiproc = time.time() - start_multiproc

    print('\n------------------\n')
    print(f'Graph size: {setup_wg.vertex_count}')
    print(f'Time default: {time_default} s')
    print(f'Time multiproc: {time_multiproc} s')
    print(f'CPUs used: {n_cpus}')
    print(f'Ratio: {time_default / time_multiproc}')
