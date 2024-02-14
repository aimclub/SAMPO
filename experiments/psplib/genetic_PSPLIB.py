import time
import uuid
import multiprocess as mp

import numpy as np

from sampo.scheduler import GeneticScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from psplib_time_estimator import PSPlibWorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.visualization import (VisualizationMode, resource_employment_fig, EmploymentFigType,
                                           schedule_gant_chart_fig)


instances = [30]
workers = ['R1', 'R2', 'R3', 'R4']


def run_scheduler(wg_info):
    adj_matrix, res_matrix, contractor_info = wg_info
    sum_of_time = res_matrix[:, 0].sum()

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

    contractor = [
        Contractor(id=str(1),
                   name="OOO Berezka",
                   workers={name: Worker(str(uuid.uuid4()), name, count, contractor_id=str(1))
                            for name, count in zip(workers, contractor_info)},
                   equipments={})
    ]

    wg = WorkGraph(nodes[0], nodes[-1])

    work_estimator = PSPlibWorkTimeEstimator(wu_id2times)

    scheduler = GeneticScheduler(20, size_of_population=50, work_estimator=work_estimator,
                                 sgs_type=ScheduleGenerationScheme.Serial, only_lft_initialization=True)
    start = time.time()
    schedule = scheduler.schedule(wg, contractor)[0]
    finish = time.time()

    # merged_schedule = schedule.merged_stages_datetime_df('2022-01-01')
    # schedule_gant_chart_fig(merged_schedule, VisualizationMode.ShowFig)
    #
    # resource_employment_fig(merged_schedule, fig_type=EmploymentFigType.DateLabeled, vis_mode=VisualizationMode.ShowFig)

    return (finish - start), schedule.execution_time, sum_of_time


if __name__ == '__main__':
    result = []
    makespans = []
    exec_times = []
    for wg_size in instances:

        with open(f'psplib_datasets/optimal_makespan/j{str(wg_size)}opt.sm', 'r') as f:
            lines = f.readlines()
            lines = lines[22:-2]
        true_val = np.array([int(line.split()[2]) for line in lines])

        dataset = np.load(f'psplib_datasets/problems_{str(wg_size)}.npy', allow_pickle=True)

        gap_sum = 0.0
        cnt = 0
        for _ in range(3):
            with mp.Pool(10) as pool:
                result = pool.starmap(run_scheduler, np.expand_dims(dataset, 1))
            result = np.array([res[1].value for res in result])

            gap = (result - true_val) / true_val * 100
            gap_sum += gap.sum()
            cnt += len(dataset)

        print(f'Average gap: {gap_sum / cnt}')
