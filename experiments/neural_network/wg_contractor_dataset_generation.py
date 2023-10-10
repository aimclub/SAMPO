import tracemalloc
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.scheduler.selection.metrics import encode_graph
from sampo.schemas.contractor import Contractor

GRAPHS_TOP_BORDER = 100
GRAPHS_COUNT = 100000
ss = SimpleSynthetic()


def display_top(snapshot, key_type='lineno', limit=3):
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


def get_resources_from_contractor(contractor: Contractor) -> list[int]:
    resources = []
    for worker in contractor.workers.values():
        resources.append(worker.count)
    return resources


def generate(index: int):
    wg = ss.work_graph(top_border=GRAPHS_TOP_BORDER)
    encoding = encode_graph(wg)
    contractor = get_contractor_by_wg(wg)
    resources = get_resources_from_contractor(contractor)
    del wg
    print('Generated')

    return tuple([encoding, resources])


if __name__ == '__main__':
    result = []
    graph_index = [[0] * (GRAPHS_COUNT // 4)] * 4
    with Pool() as pool:
        for i_graph in graph_index:
            result.extend(pool.map(generate, i_graph))
    dataset_transpose = np.array(result, dtype=object).T
    df = pd.DataFrame.from_records(dataset_transpose[0])
    df['label'] = dataset_transpose[1]
    df['label'] = df['label'].apply(lambda x: ' '.join(str(i) for i in x))
    df.fillna(value=0, inplace=True)
    # dataset_size = min(df.groupby('label', group_keys=False).apply(lambda x: len(x)))
    # df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(dataset_size))
    df.to_csv('datasets/wg_contractor_dataset_100000_objs.csv', index_label='index')
