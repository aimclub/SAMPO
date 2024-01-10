from typing import Callable

import numpy as np

from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


def get_worker_borders(agents: WorkerContractorPool, contractor: Contractor, work_reqs: list[WorkerReq]) \
        -> tuple[np.ndarray, np.ndarray, list[Worker]]:
    """
    Define for each job each type of workers the min and max possible number of workers.
    For max number of workers, max is defined as a minimum from max possible numbers
    at all and max possible for a current job.
    
    :param agents: from all projects
    :param contractor:
    :param work_reqs:
    :return:
    """
    n = len(work_reqs)
    min_worker_team = np.zeros(n, dtype=int)
    max_worker_team = np.zeros(n, dtype=int)
    workers: list[Worker] = []

    for i, req in enumerate(work_reqs):
        w = agents[req.kind].get(contractor.id, None)

        if w is None or w.count < req.min_count:
            # can't satisfy requirements, return empty to say it
            return [], [], []

        min_worker_team[i] = req.min_count
        max_worker_team[i] = min(req.max_count, w.count)
        workers.append(w)

    return min_worker_team, max_worker_team, workers


def run_contractor_search(contractors: list[Contractor],
                          runner: Callable[[Contractor], tuple[Time, Time, list[Worker]]]) \
        -> tuple[Time, Time, Contractor, list[Worker]]:
    """
    Performs the best contractor search.
    
    :param contractors: contractors' list
    :param runner: a runner function, should be inner of the calling code.
        Calculates Tuple[start time, finish time, worker team] from given contractor object.
    :return: start time, finish time, the best contractor, worker team with the best contractor
    """
    # TODO Parallelize

    # optimization metric
    best_finish_time = Time.inf()
    best_contractor = None
    # heuristic: if contractors' finish times are equal, we prefer smaller one
    best_contractor_size = float('inf')

    for contractor in contractors:
        start_time, finish_time, worker_team = runner(contractor)
        contractor_size = sum(w.count for w in contractor.workers.values())

        if not finish_time.is_inf() and (finish_time < best_finish_time or
                                         (finish_time == best_finish_time and contractor_size < best_contractor_size)):
            best_finish_time = finish_time
            best_contractor = contractor
            best_contractor_size = contractor_size

    if best_contractor is None:
        raise NoSufficientContractorError(f'There is no contractor that can satisfy given search; contractors: '
                                          f'{contractors}')

    best_start_time, best_finish_time, best_worker_team = runner(best_contractor)

    return best_start_time, best_finish_time, best_contractor, best_worker_team
