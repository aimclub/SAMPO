from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.scheduler.lft.prioritization import lft_prioritization
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.utilities.validation import validate_schedule
import pandas as pd


def test_multiprocessing(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=100,
                               mutate_order=0.05,
                               mutate_resources=0.005,
                               size_of_population=50)

    genetic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)


def test_topological(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    topological = TopologicalScheduler()

    schedule = topological.schedule(setup_wg, setup_contractors, landscape=setup_landscape)

    print(schedule.execution_time)


def test_lft_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    scheduler = HEFTScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, validate=True, landscape=setup_landscape)
    default_time = schedule.execution_time

    scheduler = HEFTScheduler(prioritization_f=lft_prioritization)
    schedule = scheduler.schedule(setup_wg, setup_contractors, validate=True, landscape=setup_landscape)
    lft_time = schedule.execution_time

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)

    except AssertionError as e:
        raise AssertionError(f'Scheduler {scheduler} failed validation', e)

    print(default_time, lft_time)


def test_lft_scheduling2(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    scheduler = HEFTBetweenScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, validate=True, landscape=setup_landscape)
    default_time = schedule.execution_time

    schedule.pure_schedule_df.to_csv('C:\\Users\\Егор\\Desktop\\lft.csv', index=False)

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)
    except AssertionError as e:
        raise AssertionError(f'Scheduler {scheduler} failed validation', e)

    print(default_time)
