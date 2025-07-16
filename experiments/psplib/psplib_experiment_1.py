import os
import time
import uuid
from random import Random

import numpy as np
import pandas as pd
import multiprocessing as mp

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler import GeneticScheduler, RandomizedTopologicalScheduler, RandomizedLFTScheduler
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


def run_basic_genetic(scheduler: GeneticScheduler,
                      wg: WorkGraph,
                      contractors: list[Contractor],
                      work_estimator: WorkTimeEstimator):
    toolbox = scheduler.create_toolbox(wg, contractors, init_schedules=[])
    pop = [randomized_init(wg, contractors, toolbox, work_estimator, i >= 25, scheduler.rand) for i in range(50)]
    final_chromosome = scheduler.upgrade_pop(wg, contractors, pop)[0]
    return toolbox.evaluate_chromosome(final_chromosome)


def randomized_init(wg: WorkGraph,
                    contractors: list[Contractor],
                    toolbox,
                    work_estimator: WorkTimeEstimator,
                    is_topological: bool = False,
                    rand: Random = Random()) -> ChromosomeType:
    if is_topological:
        seed = int(rand.random() * 1000000)
        schedule, _, _, node_order = RandomizedTopologicalScheduler(work_estimator,
                                                                    seed).schedule_with_cache(wg, contractors)[0]
    else:
        schedule, _, _, node_order = RandomizedLFTScheduler(work_estimator=work_estimator,
                                                            rand=rand).schedule_with_cache(wg, contractors)[0]
    return toolbox.schedule_to_chromosome(schedule=schedule)


def run_our_genetic(scheduler: GeneticScheduler, wg: WorkGraph, contractors: list[Contractor]):
    return scheduler.schedule(wg, contractors)[0]


def run_scheduler(args):
    adj_matrix, res_matrix, contractor_info = args
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

    contractors = [
        Contractor(id=str(1),
                   name="OOO Berezka",
                   workers={name: Worker(str(uuid.uuid4()), name, count, contractor_id=str(1))
                            for name, count in zip(workers, contractor_info)},
                   equipments={})
    ]

    wg = WorkGraph(nodes[0], nodes[-1])

    work_estimator = PSPlibWorkTimeEstimator(wu_id2times)

    scheduler = GeneticScheduler(number_of_generation=100,
                                 size_of_population=50,
                                 work_estimator=work_estimator,
                                 sgs_type=ScheduleGenerationScheme.Serial)

    # merged_schedule = schedule.merged_stages_datetime_df('2022-01-01')
    # schedule_gant_chart_fig(merged_schedule, VisualizationMode.ShowFig)
    #
    # resource_employment_fig(merged_schedule, fig_type=EmploymentFigType.DateLabeled, vis_mode=VisualizationMode.ShowFig)

    start_basic = time.time()
    schedule_basic = run_basic_genetic(scheduler, wg, contractors, work_estimator)
    finish_basic = time.time()
    results_basic = (finish_basic - start_basic), schedule_basic.execution_time, sum_of_time, schedule_basic

    start_our = time.time()
    schedule_our = run_our_genetic(scheduler, wg, contractors)
    finish_our = time.time()
    results_our = (finish_our - start_our), schedule_our.execution_time, sum_of_time, schedule_our

    return results_basic, results_our


if __name__ == '__main__':
    result = []
    makespans = []
    exec_times = []
    attempts = 10

    os.makedirs('experiment_results', exist_ok=True)
    os.makedirs('experiment_results/schedules', exist_ok=True)
    os.makedirs('experiment_results/basic', exist_ok=True)
    os.makedirs('experiment_results/our', exist_ok=True)

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

            for wg_idx, ((res_basic, res_our), val) in enumerate(zip(result, true_val)):
                def make_result(res, type):
                    exec_time, makespan, psplib_time, schedule = res

                    peak_resource_usage = resources_peaks_sum(schedule)

                    schedule_file_name = f'{wg_size}_{attempt}_{wg_idx}'

                    schedule.dump(f'experiment_results/schedules/{type}', schedule_file_name)

                    results.append((type, wg_size, attempt, wg_idx, val, exec_time, makespan, peak_resource_usage))

                make_result(res_basic, 'basic')
                make_result(res_our, 'our')

        pd.DataFrame.from_records(results, columns=['type', 'wg_size', 'attempt', 'wg_idx', 'true_val', 'exec_time',
                                                    'makespan', 'peak_resource_usage']).to_csv(f'experiment_results/psplib_experiment_j{wg_size}_results.csv')
        results = []
