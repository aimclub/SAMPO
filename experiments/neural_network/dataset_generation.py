from multiprocessing import Pool

import numpy as np
import pandas as pd

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.selection.metrics import encode_graph
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.time import Time

GRAPHS_TOP_BORDER = 200
GRAPHS_COUNT = 10000
ss = SimpleSynthetic()

contractors = [ss.contractor(10)]
schedulers = [HEFTScheduler(), HEFTBetweenScheduler(), TopologicalScheduler()]


def argmin(array) -> int:
    res = 0
    res_v = int(Time.inf())
    for i, v in enumerate(array):
        if v < res_v:
            res_v = v
            res = i
    return res


# def generate_graphs(labels_count: int, bin_size: int) -> list[tuple[list[float], int]]:
#     bins = [0 for _ in range(labels_count)]
#     result = []
#
#     while any((bin < bin_size for bin in bins)):
#         wg = ss.work_graph(top_border=GRAPHS_TOP_BORDER)
#         encoding = encode_graph(wg)
#         schedulers_results = [int(scheduler.schedule(wg, contractors).execution_time) for scheduler in schedulers]
#         generated_label = argmin(schedulers_results)
#
#         if bins[generated_label] < bin_size:
#             bins[generated_label] += 1
#             result.append((encoding, generated_label))
#             if bins[generated_label] % 10 == 0:
#                 print(f'{generated_label}: {bins[generated_label]}/{bin_size} processed')
#     return result

def generate_graph(label: int):
    while True:
        try:
            wg = ss.work_graph(top_border=GRAPHS_TOP_BORDER)
            encoding = encode_graph(wg)
            schedulers_results = [int(scheduler.schedule(wg, contractors).execution_time) for scheduler in schedulers]
            generated_label = argmin(schedulers_results)

            if generated_label == label:
                print(f'{generated_label} processed')
                return tuple([encoding, generated_label])
        except Exception:
            raise Exception()


if __name__ == '__main__':
    result = []
    with Pool() as pool:
        for i_scheduler in range(len(schedulers)):
            tasks = [[i_scheduler] * int(GRAPHS_COUNT / 4)] * 4
            for task in tasks:
                result.extend(pool.map(generate_graph, task))

    dataset_transposed = np.array(result).T
    df = pd.DataFrame.from_records(dataset_transposed[0])
    df['label'] = dataset_transposed[1]
    df.fillna(value=0, inplace=True)
    dataset_size = min(df.groupby('label', group_keys=False).apply(lambda x: len(x)))
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(dataset_size))
    df.to_csv('dataset.csv')
