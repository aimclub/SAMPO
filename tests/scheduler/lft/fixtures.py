from pytest import fixture

from sampo.scheduler.lft.base import LFTScheduler, RandomizedLFTScheduler
import random


@fixture(params=[LFTScheduler, RandomizedLFTScheduler],
         ids=[f'Scheduler: {scheduler}' for scheduler in ['LFTScheduler', 'RandomizedLFTScheduler']])
def setup_schedulers_and_parameters(request, setup_scheduler_parameters) -> tuple:
    scheduler = request.param
    if isinstance(scheduler, RandomizedLFTScheduler):
        scheduler = scheduler(rand=random.Random(2023))
    else:
        scheduler = scheduler()

    setup_wg, setup_contractors, setup_landscape, spec, rand = setup_scheduler_parameters

    return setup_wg, setup_contractors, setup_landscape, spec, rand, scheduler
