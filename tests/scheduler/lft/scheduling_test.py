from sampo.utilities.validation import validate_schedule
from tests.scheduler.lft.fixtures import setup_schedulers_and_parameters


def test_lft_scheduling(setup_schedulers_and_parameters):
    setup_wg, setup_contractors, setup_landscape, spec, rand, scheduler = setup_schedulers_and_parameters

    schedule = scheduler.schedule(setup_wg, setup_contractors,
                                  spec=spec,
                                  validate=True,
                                  landscape=setup_landscape)[0]
    lft_time = schedule.execution_time

    assert not lft_time.is_inf()

    try:
        validate_schedule(schedule, setup_wg, setup_contractors)
    except AssertionError as e:
        raise AssertionError(f'Scheduler {scheduler} failed validation', e)
