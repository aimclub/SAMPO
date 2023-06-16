import pytest

from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.utilities.validation import validate_schedule


def test_just_in_time_scheduling_with_materials(setup_default_schedules, setup_landscape):
    setup_wg, setup_contractors = setup_default_schedules[0]
    if setup_wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    scheduler = HEFTScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, setup_landscape, validate=True)


def test_momentum_scheduling_with_materials(setup_default_schedules, setup_landscape_with_many_holders):
    setup_wg, setup_contractors = setup_default_schedules[0]
    if setup_wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    scheduler = HEFTBetweenScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, setup_landscape_with_many_holders, validate=True)

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)

    except AssertionError as e:
        raise AssertionError(f'Scheduler {scheduler} failed validation', e)


def test_scheduler_with_materials_validity_right(setup_schedule):
    schedule = setup_schedule[0]
    setup_wg, setup_contractors = setup_schedule[2]

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)
    except AssertionError as e:
        raise AssertionError(f'Scheduler {setup_schedule[1]} failed validation', e)
