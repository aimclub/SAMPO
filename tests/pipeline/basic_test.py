import pytest

from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.exceptions import NoSufficientContractorError


def test_plain_scheduling(setup_wg, setup_contractors):
    try:
        schedule = SchedulingPipeline.create() \
            .wg(setup_wg) \
            .contractors(setup_contractors) \
            .schedule(HEFTScheduler()) \
            .finish()

        print(f'Scheduled {len(schedule.to_schedule_work_dict)} works')
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')
