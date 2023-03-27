from uuid import uuid4

import pytest
from pytest import fixture

from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.landscape import LandscapeConfiguration, ResourceHolder
from sampo.schemas.resources import Material


@fixture(scope='function')
def setup_landscape():
    return LandscapeConfiguration(holders=[ResourceHolder(str(uuid4()), 'holder1', IntervalGaussian(25, 0),
                                                          materials=[Material('111', 'mat1', 100000)])])


def test_scheduling_with_materials(setup_wg, setup_contractors, setup_landscape):
    if setup_wg.vertex_count > 14:
        pytest.skip('Non-material graph')

    scheduler = HEFTScheduler()
    schedule = scheduler.schedule(setup_wg, setup_contractors, setup_landscape, validate=True)

    pass
