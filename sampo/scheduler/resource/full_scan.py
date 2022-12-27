from typing import List, Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.utilities.base_opt import dichotomy_int


class FullScanResourceOptimizer(ResourceOptimizer):

    _coordinate_descent_optimizer = CoordinateDescentResourceOptimizer(dichotomy_int)

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: List[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):

        # TODO Handle optimize_array
        def fitness(worker_count: np.ndarray):
            for worker_ind in range(len(worker_team)):
                worker_team[worker_ind].count = worker_count[worker_ind]
            return get_finish_time(worker_team)

        cur = down_border.copy()
        cur_ft = fitness(cur)

        # Trying to +1 to all workers
        while (up_border > cur).all():
            cur += 1
            next_ft = fitness(cur)
            if next_ft >= cur_ft:
                cur -= 1
                break

        # Insert cur into coordinate-descent optimizer as down_border
        self._coordinate_descent_optimizer.optimize_resources(worker_pool,
                                                              worker_team,
                                                              optimize_array,
                                                              cur,
                                                              up_border,
                                                              get_finish_time)
