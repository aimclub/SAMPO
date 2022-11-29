import pytest

from sampo.scheduler.base import SchedulerType


@pytest.mark.parametrize('setup_schedule', list(SchedulerType), indirect=True)
def test_schedule_object(setup_schedule):
    schedule, scheduler = setup_schedule

    full_df = schedule.full_schedule_df
    pure_df = schedule.pure_schedule_df
    s_works = list(schedule.works)
    swd = schedule.to_schedule_work_dict
    exec_time = schedule.execution_time

    print(f'{scheduler} schedule is OK')
