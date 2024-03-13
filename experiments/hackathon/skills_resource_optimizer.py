from collections import defaultdict
from operator import attrgetter
from typing import Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas import Worker, Time


class GreedyMinimalMultiSkillResourceOptimizer(ResourceOptimizer):
    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: list[Worker],
                           optimize_array: np.ndarray | None,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[list[Worker]], Time]):
        # separate workers by specialization
        bins = defaultdict(list)
        for worker in worker_team:
            bins[worker.name.split(' ')[0]].append(worker)
            worker.count = 0

        # for bin_ in bins:
        #     bin_.sort(key=attrgetter('cost_one_unit'))

        for bin_ in bins.values():
            min_cost_worker = bin_[0]
            for worker in bin_[1:]:
                if worker.cost_one_unit < min_cost_worker.cost_one_unit:
                    min_cost_worker = worker
            min_cost_worker.count = 1
