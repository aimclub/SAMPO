import pytest

from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer, ParallelizeScheduleLocalOptimizer
from sampo.schemas.exceptions import NoSufficientContractorError


def test_plain_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors = setup_scheduler_parameters
    try:
        schedule = SchedulingPipeline.create() \
            .wg(setup_wg) \
            .contractors(setup_contractors) \
            .optimize_local(SwapOrderLocalOptimizer(), range(0, setup_wg.vertex_count // 2)) \
            .schedule(HEFTScheduler()) \
            .optimize_local(ParallelizeScheduleLocalOptimizer(JustInTimeTimeline), range(0, setup_wg.vertex_count // 2)) \
            .finish()

        print(f'Scheduled {len(schedule.to_schedule_work_dict)} works')
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')
