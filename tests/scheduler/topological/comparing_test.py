from datetime import time
from time import time
from typing import Optional

import pytest

from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.time_estimator import WorkTimeEstimator


def test_comparing_to_heft(setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters

    work_estimator: Optional[WorkTimeEstimator] = None

    def init_schedule(scheduler_class):
        return scheduler_class(work_estimator=work_estimator).schedule(setup_wg, setup_contractors, landscape=landscape)

    try:
        start_time = time()
        init_schedule(HEFTScheduler)
        heft_time = time() - start_time
        start_time = time()
        init_schedule(TopologicalScheduler)
        topological_time = time() - start_time

        print(f'HEFT time: {heft_time * 1000} ms')
        print(f'Topological time: {topological_time * 1000} ms')

        winner = 'HEFT' if heft_time < topological_time else 'Topological'
        ratio = max(heft_time, topological_time) / max(min(heft_time, topological_time), 0.01)

        print(f'{winner} wins with coefficient {ratio}!')
    except NoSufficientContractorError:
        pytest.skip('Given contractor configuration can\'t support given work graph')

