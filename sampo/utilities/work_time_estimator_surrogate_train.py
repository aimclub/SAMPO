import multiprocessing as mp
from typing import Callable, Optional
from uuid import uuid4

import numpy as np

from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit


def init_worker(wte_constructor, wte_args):
    global wte
    wte = wte_constructor(*wte_args)


def calculate_time(args: tuple[list[int], list[str], list[IntervalGaussian | None]]) -> int:
    counts, names, productivities = args
    work_unit = WorkUnit("", "")
    contractor = str(uuid4())
    worker_team = [Worker(str(uuid4()), name, count, contractor, productivity)
                   for name, count, productivity in zip(names, counts, productivities)]
    return work_unit.estimate_static(worker_team, wte).value


# gets the 'next' team
def inc(team: list[int], max_counts: list[int]):
    i = len(team) - 1
    while i >= 0:
        team[i] += 1
        if team[i] > max_counts[i]:
            team[i] = 0
            i -= 1
        else:
            break


def calculate_arg_results(worker_labels: list[str],
                          worker_counts: list[int],
                          work_time_estimator: WorkTimeEstimator):
    n_workers = len(worker_labels)
    iterations = 1  # teams count
    for v in worker_counts:
        if v != 0:
            iterations *= v + 1

    count_args = []
    cur_team = [0 for _ in range(n_workers)]
    productivities = [None for _ in range(n_workers)]

    for i in range(iterations):
        count_args.append((cur_team[:], worker_labels, productivities))
        inc(cur_team, worker_counts)

    with mp.Pool(initializer=init_worker,
                 initargs=work_time_estimator.split_to_constructor_and_params()) as p:
        results = p.map(calculate_time, count_args)

        with open('trained.txt', 'w') as f:
            for res, count_arg in zip(results, count_args):
                cur_team = count_arg[0]
                f.write(' '.join(map(str, cur_team)))
                f.write(f' {res}')
                f.write('\n')


class TestWorkTimeEstimator(WorkTimeEstimator):

    def set_mode(self, use_idle: Optional[bool] = True, mode: Optional[str] = 'realistic'):
        pass

    def estimate_time(self, work_name: str, work_volume: float, resources: WorkerContractorPool) -> Time:
        return Time(int(work_volume))

    def split_to_constructor_and_params(self) -> tuple[Callable[[...], 'WorkTimeEstimator'], tuple]:
        return TestWorkTimeEstimator, ()


def get(teams: np.ndarray, team: list[int]) -> int:
    cur = teams

    for next_worker in team:
        cur = cur[next_worker]

    return cur


def train_surrogate(teams: np.ndarray):
    a1 = a2 = a3 = a4 = 1




if __name__ == '__main__':
    worker_labels = ['1', '2', '3', '4', '5']
    worker_counts = [ 1 ,  1 ,  1 ,  1 ,  1 ]

    calculate_arg_results(worker_labels, worker_counts, TestWorkTimeEstimator())
