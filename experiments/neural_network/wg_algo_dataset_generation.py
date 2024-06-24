import tracemalloc
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.selection.metrics import encode_graph
from sampo.schemas.time import Time

GRAPHS_TOP_BORDER = 100
GRAPHS_COUNT = 10000
ss = SimpleSynthetic()

contractors = [ss.contractor(10)]
schedulers = [HEFTScheduler(), HEFTBetweenScheduler()]


def argmin(array) -> int:
    res = 0
    res_v = int(Time.inf())
    for i, v in enumerate(array):
        if v < res_v:
            res_v = v
            res = i
    return res


def display_top(snapshot, key_type='lineno', limit=3):
    """
    For tracking the volume of RAM used
    """
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def generate() -> tuple:
    wg = ss.work_graph(top_border=GRAPHS_TOP_BORDER)
    encoding = encode_graph(wg)
    schedulers_results = [int(scheduler.schedule(wg, contractors)[0].execution_time) for scheduler in schedulers]
    generated_label = argmin(schedulers_results)
    del wg
    del schedulers_results

    return generated_label, encoding


def generate_graph(label: int) -> tuple:
    while True:
        # Uncomment for tracking the volume of RAM used
        # tracemalloc.start()
        generated_label, encoding = generate()
        if generated_label == label:
            # Uncomment for tracking the volume of RAM used
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)
            print(f'{generated_label} processed')
            return tuple([encoding, generated_label])


if __name__ == '__main__':
    result = []
    with Pool() as pool:
        for i_scheduler in range(len(schedulers)):
            # int(CRAPH_COUNT / 4) - number of parallel processes
            tasks = [[i_scheduler] * int(GRAPHS_COUNT / 4)] * 4
            for task in tasks:
                result.extend(pool.map(generate_graph, task))

    dataset_transposed = np.array(result, dtype=object).T
    df = pd.DataFrame.from_records(dataset_transposed[0])
    df['label'] = dataset_transposed[1]
    df.fillna(value=0, inplace=True)
    dataset_size = min(df.groupby('label', group_keys=False).apply(lambda x: len(x)))
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(dataset_size))
    df.to_csv('datasets/wg_algo_dataset_10k.csv', index_label='index')
