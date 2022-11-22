from datetime import time
from time import time
from typing import Optional

from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler


def test_comparing_to_heft(setup_wg, setup_contractors, setup_start_date):
    work_estimator: Optional[WorkTimeEstimator] = None

    def init_schedule(scheduler_class):
        return scheduler_class(work_estimator).schedule(setup_wg, setup_contractors, setup_start_date)

    start_time = time()
    init_schedule(HEFTScheduler)
    heft_time = time() - start_time
    start_time = time()
    init_schedule(TopologicalScheduler)
    topological_time = time() - start_time

    print(f'HEFT time: {heft_time * 1000} ms')
    print(f'Topological time: {topological_time * 1000} ms')

    winner = 'HEFT' if heft_time < topological_time else 'Topological'
    ratio = max(heft_time, topological_time) / min(heft_time, topological_time)

    print(f'{winner} wins with coefficient {ratio}!')

