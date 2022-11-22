from typing import List

import numpy as np

from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker


def get_best_contractor_and_worker_borders(agents: WorkerContractorPool,
                                           contractors: List[Contractor],
                                           work_reqs: List[WorkerReq]) \
        -> (np.ndarray, np.ndarray, Contractor, List[Worker]):
    """
    Define for each job each type of workers the min and max possible number of workers
    For max number of workers max is define as minimum from max possible numbers at all and max possible for current job
    :param agents: from all project
    :param contractors:
    :param work_reqs:
    :param timeline:
    :return:
    """
    n = len(work_reqs)
    min_worker_team = np.zeros(n, dtype=int)
    max_worker_team = np.zeros(n, dtype=int)
    workers: List[Worker] = []
    contractor: Contractor = contractors[0]
    min_sum = 1e50

    # TODO Entry point for multi-contractor
    # TODO When starting implementation of multi-contractor,
    #  see this algo about searching the most relevant contractor
    for c in contractors:
        cur_min_worker_team = np.zeros(n, dtype=int)
        cur_max_worker_team = np.zeros(n, dtype=int)
        cur_workers = []
        satisfied = True

        for i, req in enumerate(work_reqs):
            offers = agents[req.kind]

            cur_offers = list(filter(lambda x: x.contractor_id == c.id, offers.values()))
            if len(cur_offers) == 0 or cur_offers[0].count < req.min_count:
                satisfied = False
                break

            cur_min_worker_team[i] = req.min_count
            cur_max_worker_team[i] = min(req.max_count, cur_offers[0].count)
            cur_workers.append(cur_offers[0])

        cur_sum = np.sum(cur_max_worker_team - cur_min_worker_team)
        if satisfied and cur_sum < min_sum:
            min_worker_team = cur_min_worker_team
            max_worker_team = cur_max_worker_team
            workers = cur_workers
            min_sum = cur_sum
            contractor = c

    return min_worker_team, max_worker_team, contractor, workers
