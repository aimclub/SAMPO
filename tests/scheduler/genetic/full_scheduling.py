import time

from sampo.api.genetic_api import ScheduleGenerationScheme
from sampo.backend.default import DefaultComputationalBackend
from sampo.backend.multiproc import MultiprocessingComputationalBackend
from sampo.backend.native import NativeComputationalBackend
from sampo.base import SAMPO
from sampo.scheduler import GeneticScheduler
from sampo.scheduler.timeline import JustInTimeTimeline
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.schemas.schedule_spec import ScheduleSpec


def test_multiprocessing(setup_scheduler_parameters):
    wg, contractors, landscape = setup_scheduler_parameters

    SAMPO.backend = DefaultComputationalBackend()

    genetic = GeneticScheduler(number_of_generation=10,
                               mutate_order=0.05,
                               mutate_resources=0.05,
                               size_of_population=50,
                               sgs_type=ScheduleGenerationScheme.Parallel)

    start_default = time.time()
    genetic.schedule(wg, contractors, validate=True, landscape=landscape,
                     timeline=JustInTimeTimeline(get_worker_contractor_pool(contractors), landscape))
    time_default = time.time() - start_default

    n_cpus = 10
    SAMPO.backend = NativeComputationalBackend()

    start_multiproc = time.time()
    genetic.schedule(wg, contractors, landscape=landscape)
    time_multiproc = time.time() - start_multiproc

    print('\n------------------\n')
    print(f'Graph size: {wg.vertex_count}')
    print(f'Time default: {time_default} s')
    print(f'Time multiproc: {time_multiproc} s')
    print(f'CPUs used: {n_cpus}')
    print(f'Ratio: {time_default / time_multiproc}')
