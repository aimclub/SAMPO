def test_schedule_object(setup_schedule):
    schedule, scheduler = setup_schedule

    # check that main properties are not failing to access
    full_df = schedule.full_schedule_df
    pure_df = schedule.pure_schedule_df
    s_works = list(schedule.works)
    swd = schedule.to_schedule_work_dict

    assert not schedule.execution_time.is_inf(), f'Scheduling failed on {scheduler}'
