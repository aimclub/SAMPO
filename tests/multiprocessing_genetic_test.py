import pytest

from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.schemas.exceptions import NoSufficientContractorError


def test_genetic_run(setup_wg, setup_contractors):
    genetic = GeneticScheduler()
    try:
        genetic.schedule(setup_wg, setup_contractors)
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')
