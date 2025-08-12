import os
import time
import uuid
from random import Random

import numpy as np
import pandas as pd
import multiprocessing as mp

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler import GeneticScheduler, RandomizedTopologicalScheduler, RandomizedLFTScheduler, HEFTScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme
from sampo.schemas import WorkTimeEstimator
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from psplib_time_estimator import PSPlibWorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.resource_usage import resources_peaks_sum


instances = [30, 60, 90, 120]
workers = ['R1', 'R2', 'R3', 'R4']


def run_scheduler(args):
    adj_matrix, res_matrix, contractor_info = args

    nodes = []
    wu_id2times = {}

    for node_id, res_matrix_row in enumerate(res_matrix):
        worker_reqs = [WorkerReq(workers[idx_req - 1], Time(0), res_matrix_row[idx_req],
                                 res_matrix_row[idx_req], workers[idx_req - 1]) for idx_req in range(1, 5)]
        work_unit = WorkUnit(str(node_id), str(node_id), worker_reqs)
        wu_id2times[work_unit.id] = Time(int(res_matrix_row[0]))
        node = GraphNode(work_unit, [])
        nodes.append(node)

    for child_id in range(len(adj_matrix)):
        parents = []
        for parent_id in range(len(adj_matrix)):
            if child_id != parent_id and adj_matrix[parent_id][child_id] == 1:
                parents.append((nodes[parent_id], 0, EdgeType.FinishStart))
        nodes[child_id].add_parents(parents)

    contractors = [
        Contractor(id=str(1),
                   name="OOO Berezka",
                   workers={name: Worker(str(uuid.uuid4()), name, 10_000_000_000, contractor_id=str(1))
                            for name, count in zip(workers, contractor_info)},
                   equipments={})
    ]

    wg = WorkGraph(nodes[0], nodes[-1])

    work_estimator = PSPlibWorkTimeEstimator(wu_id2times)

    scheduler = HEFTScheduler(work_estimator=work_estimator)

    schedule_infinite = scheduler.schedule(wg, contractors)[0]

    return schedule_infinite.execution_time


if __name__ == '__main__':
    result = []
    makespans = []
    exec_times = []

    os.makedirs('experiment_results', exist_ok=True)
    os.makedirs('experiment_results/schedules', exist_ok=True)
    os.makedirs('experiment_results/schedules/basic', exist_ok=True)
    os.makedirs('experiment_results/schedules/our', exist_ok=True)

    attempts = 10
    results = []

    for wg_size in instances:
        with open(f'psplib_datasets/optimal_makespan/j{str(wg_size)}opt.sm', 'r') as f:
            lines = f.readlines()
            lines = lines[22:-2]
        true_val = np.array([int(line.split()[2]) for line in lines])

        dataset = np.load(f'psplib_datasets/problems_{str(wg_size)}.npy', allow_pickle=True)

        for attempt in range(attempts):
            with mp.Pool(32) as pool:
                result = pool.starmap(run_scheduler, np.expand_dims(dataset, 1))

            for wg_idx, (psplib_time, val) in enumerate(zip(result, true_val)):
                results.append((wg_size, wg_idx, attempt, psplib_time))

        pd.DataFrame.from_records(results, columns=['wg_size', 'wg_idx', 'attempt', 'psplib_time']) \
                                            .to_csv(f'experiment_results/psplib_lb_j{wg_size}.csv')
        results = []
