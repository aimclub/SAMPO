import os
import sys

from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer, ParallelizeScheduleLocalOptimizer
from sampo.schemas.exceptions import NoSufficientContractorError


def test_plain_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape, spec, rand = setup_scheduler_parameters

    project = SchedulingPipeline.create() \
        .wg(setup_wg) \
        .contractors(setup_contractors) \
        .landscape(setup_landscape) \
        .spec(spec) \
        .schedule(HEFTScheduler()) \
        .finish()[0]

    print(f'Scheduled {len(project.schedule.to_schedule_work_dict)} works')


def test_local_optimize_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape, spec, rand = setup_scheduler_parameters

    project = SchedulingPipeline.create() \
        .wg(setup_wg) \
        .contractors(setup_contractors) \
        .landscape(setup_landscape) \
        .spec(spec) \
        .optimize_local(SwapOrderLocalOptimizer(), range(0, setup_wg.vertex_count // 2)) \
        .schedule(HEFTScheduler()) \
        .optimize_local(ParallelizeScheduleLocalOptimizer(JustInTimeTimeline), range(0, setup_wg.vertex_count // 2)) \
        .finish()[0]

    print(f'Scheduled {len(project.schedule.to_schedule_work_dict)} works')


# this test is needed to check validation of input contractors

def test_plain_scheduling_with_no_sufficient_number_of_contractors(setup_wg, setup_empty_contractors):
    thrown = False
    if setup_wg.vertex_count > 16:
        pass
    try:
        SchedulingPipeline.create() \
            .wg(setup_wg) \
            .contractors(setup_empty_contractors) \
            .schedule(HEFTScheduler()) \
            .finish()
    except NoSufficientContractorError:
        thrown = True

    assert thrown


def test_plain_scheduling_with_parse_data():
    wg = os.path.join(sys.path[0], 'tests/parser/test_wg.csv')

    project = SchedulingPipeline.create() \
        .wg(wg=wg, sep=';', all_connections=True) \
        .lag_optimize(LagOptimizationStrategy.TRUE) \
        .schedule(HEFTScheduler()) \
        .finish()[0]

    schedule = project.schedule
    schedule = schedule.merged_stages_datetime_df('2022-01-01')
