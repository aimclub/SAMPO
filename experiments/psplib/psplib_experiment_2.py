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


instances = [120]
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
    filename = args
    with open(filename, 'r') as f:
        wg, contractors, wu_id2times, _ = parse_sm_file(f.readlines())

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
    results_basic = (finish_basic - start_basic), schedule_basic.execution_time, schedule_basic

    start_our = time.time()
    schedule_our = run_our_genetic(scheduler, wg, contractors)
    finish_our = time.time()
    results_our = (finish_our - start_our), schedule_our.execution_time, schedule_our

    return results_basic, results_our


def read_sm_line(line: str) -> list[int]:
    return list(map(int, line.split()))


def parse_sm_file(file: list[str]) -> tuple[WorkGraph, list[Contractor], dict[str, int], int]:
    tasks_count = int(file[5].split(':')[1].strip())
    project_duration = int(file[14].split()[3])
    resources_count = 4

    PRECEDENCE_START = 18

    tasks = []
    wu_id2times = {}

    # read resources
    for i in range(PRECEDENCE_START + tasks_count + 4, PRECEDENCE_START + tasks_count + 4 + tasks_count):
        job_index, _, duration, *resources = read_sm_line(file[i])
        wu_id = str(uuid.uuid4())
        tasks.append(GraphNode(WorkUnit(id=wu_id,
                                        name=f'Task #{job_index}',
                                        worker_reqs=[WorkerReq(kind=f'R{j}',
                                                               volume=Time(0),
                                                               min_count=resources[j],
                                                               max_count=resources[j])
                                                     for j in range(resources_count)
                                                     if resources[j] > 0]),
                               parent_works=[]))
        wu_id2times[wu_id] = duration

    # read precedence
    for i in range(PRECEDENCE_START, PRECEDENCE_START + tasks_count):
        job_index, _, _, *successors = read_sm_line(file[i])
        for successor in successors:
            tasks[successor - 1].add_parents([tasks[job_index - 1]])

    # read contractor
    contractor_info = list(map(int, file[PRECEDENCE_START + tasks_count + tasks_count + 4 + 3].split()))
    contractor = Contractor(id=str(uuid.uuid4()),
                            name='OOO Berezka',
                            workers={f'R{j}': Worker(id=str(uuid.uuid4()),
                                                     name=f'R{j}',
                                                     count=contractor_info[j])
                                     for j in range(resources_count)})
    return WorkGraph.from_nodes(tasks), [contractor], wu_id2times, project_duration


if __name__ == '__main__':
    result = []
    makespans = []
    exec_times = []
    attempts = 10

    os.makedirs('experiment_results', exist_ok=True)
    os.makedirs('experiment_results/schedules', exist_ok=True)
    os.makedirs('experiment_results/schedules/basic', exist_ok=True)
    os.makedirs('experiment_results/schedules/our', exist_ok=True)

    results = []

    for wg_size in instances:
        with open(f'psplib_datasets/optimal_makespan/j{str(wg_size)}opt.sm', 'r') as f:
            lines = f.readlines()
            lines = lines[22:-2]
        true_val = np.array([int(line.split()[2]) for line in lines])

        def psplib_index_comparator(a: str):
            a = a[len(str(wg_size)) + 1:-3]
            a_tuple = list(map(int, a.split('_')))
            return a_tuple

        dataset = []
        for file in sorted(os.listdir(f'psplib_datasets/j{wg_size}.sm'), key=psplib_index_comparator)[:len(true_val) + 1]:
            filename = f'psplib_datasets/j{wg_size}.sm/{file}'
            dataset.append(filename)

        for attempt in range(attempts):
            with mp.Pool(10) as pool:
                result = pool.map(run_scheduler, dataset[:10])

            for wg_idx, ((res_basic, res_our), val) in enumerate(zip(result, true_val)):
                def make_result(res, type):
                    exec_time, makespan, schedule = res

                    peak_resource_usage = resources_peaks_sum(schedule)

                    schedule_file_name = f'{wg_size}_{attempt}_{wg_idx}'

                    schedule.dump(f'experiment_results/schedules/{type}', schedule_file_name)

                    results.append((type, wg_size, attempt, wg_idx, val, exec_time, makespan, peak_resource_usage))

                make_result(res_basic, 'basic')
                make_result(res_our, 'our')

        pd.DataFrame.from_records(results, columns=['type', 'wg_size', 'attempt', 'wg_idx', 'true_val', 'exec_time',
                                                    'makespan', 'peak_resource_usage']).to_csv(f'experiment_results/psplib_experiment_j{wg_size}_results.csv')
        results = []
