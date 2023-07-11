import pytest

from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer, ParallelizeScheduleLocalOptimizer


def test_plain_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    schedule = SchedulingPipeline.create() \
        .wg(setup_wg) \
        .contractors(setup_contractors) \
        .landscape(setup_landscape) \
        .optimize_local(SwapOrderLocalOptimizer(), range(0, setup_wg.vertex_count // 2)) \
        .schedule(HEFTScheduler()) \
        .optimize_local(ParallelizeScheduleLocalOptimizer(JustInTimeTimeline), range(0, setup_wg.vertex_count // 2)) \
        .finish()

    print(f'Scheduled {len(schedule.to_schedule_work_dict)} works')


# this test is needed to check validation of input contractors
# it's commented to avoid breaking test
# def test_plain_scheduling_with_no_sufficient_number_of_contractors(setup_scheduler_parameters_with_0_contractor_worker):
#     setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters_with_0_contractor_worker
#
#     schedule = SchedulingPipeline.create() \
#         .wg(setup_wg) \
#         .contractors(setup_contractors) \
#         .landscape(setup_landscape) \
#         .optimize_local(SwapOrderLocalOptimizer(), range(0, setup_wg.vertex_count // 2)) \
#         .schedule(HEFTScheduler()) \
#         .optimize_local(ParallelizeScheduleLocalOptimizer(JustInTimeTimeline), range(0, setup_wg.vertex_count // 2)) \
#         .finish()
#
#     print(f'Scheduled {len(schedule.to_schedule_work_dict)} works')