import multiprocessing as mp
from uuid import uuid4

from sampo.schemas.interval import IntervalGaussian
from sampo.schemas.resources import Worker
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
            team[i - 1] += 1


def train_surrogate(worker_labels: list[str],
                    worker_counts: list[int],
                    work_time_estimator: WorkTimeEstimator):
    n_workers = len(worker_labels)
    iterations = 1  # teams count
    for v in worker_counts:
        if v != 0:
            iterations *= v

    count_args = []
    cur_team = [0 for _ in range(n_workers)]
    productivities = [None for _ in range(n_workers)]

    for i in range(iterations):
        count_args.append((cur_team, worker_labels, productivities))
        inc(cur_team, worker_counts)

    with mp.Pool(initializer=init_worker,
                 initargs=work_time_estimator.split_to_constructor_and_params()) as p:
        p.map(calculate_time, count_args)
