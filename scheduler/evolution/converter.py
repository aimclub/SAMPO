from typing import Dict, List, Tuple, Union
from uuid import uuid4

import numpy as np

from schemas.schedule import ScheduledWork, ScheduleWorkDict, Schedule
from schemas.work_estimator import WorkTimeEstimator
from schemas.contractor import AgentsDict
from schemas.resources import Worker
from schemas.time import Time
from schemas.works import WorkUnit
from schemas.graph import GraphNode

ChromosomeType = Tuple[List[int], np.ndarray]


def convert_schedule_to_chromosome(index2node: Dict[int, GraphNode],
                                   work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                                   schedule: Union[Schedule, ScheduleWorkDict], order: List[str]) \
        -> Tuple[List[int], np.ndarray]:
    """
    received result of HEFT algorithm and transform it to chromosome
    :param work_id2index:
    :param order:
    :param schedule:
    :param index2node:
    :param worker_name2index:
    :return:
    """
    # TODO: consider employing Schedule without converting to ScheduleWorkDict
    # convert to convenient form
    schedule = schedule.to_schedule_work_dict \
        if isinstance(schedule, Schedule) \
        else schedule

    # order works part of chromosome
    order_chromosome: List[int] = [work_id2index[work_id] for work_id in order]

    # resources for works part of chromosome
    resource_chromosome = np.zeros((len(worker_name2index), len(order_chromosome)), dtype=int)

    for index, node in index2node.items():
        for resource in schedule[node.id].workers:
            if resource in node.work_unit.worker_reqs:
                res_count = resource.count
                res_index = worker_name2index[resource.name]
                work_index = work_id2index[node.id]
                resource_chromosome[res_index, work_index] = res_count

    chromosome: Tuple[List[int], np.ndarray] = (order_chromosome, resource_chromosome)
    return chromosome


def convert_chromosome_to_schedule(chromosome: ChromosomeType, agents: AgentsDict, index2node: Dict[int, GraphNode],
                                   work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                                   work_estimator: WorkTimeEstimator = None) -> Dict[str, ScheduledWork]:
    scheduled_work: Dict[str, ScheduledWork] = {}

    time_resources_queue: List[Tuple[Time, Dict[str, Worker]]] = [*zip((0,) * len(agents), *zip(*agents.items()))]
    works_order = chromosome[0]
    works_resources = chromosome[1]
    for index in works_order:
        if index2node[index].id in scheduled_work:
            continue
        resources = works_resources[:, index]
        worker_team: List[Worker] = [Worker(str(uuid4()), worker_name, resources[worker_index])
                                     for worker_name, worker_index in worker_name2index.items()
                                     if resources[worker_index] > 0]

        start_time, time_resources_queue = \
            find_start_time(index2node[index], worker_team, scheduled_work, time_resources_queue)
        finish_time = find_finish_time(start_time, index, index2node, work_id2index, worker_team,
                                       scheduled_work, work_estimator)

        time_resources_queue += [(finish_time, worker.name, worker.count) for worker in worker_team]
        time_resources_queue = list(filter(lambda x: len(x[1]) > 0, time_resources_queue))
        time_resources_queue = sorted(time_resources_queue)
    return scheduled_work


def find_start_time(node: GraphNode, worker_team: List[Worker],
                    scheduled_work: Dict[str, ScheduledWork],
                    time_resources_queue: List[Tuple[Time, Dict[str, Worker]]]) \
        -> Tuple[Time, List[Tuple[Time, Dict[str, Worker]]]]:
    start_time = max([scheduled_work[edges.start.id].start_end_time[1] for edges in node.edges_to] + [Time(0)])
    needed_workers = {worker.name: worker.count for worker in worker_team}
    all_needed_workers = sum(needed_workers.values(), 0)
    next_time_resources_queue: List[Tuple[Time, Dict[str, Worker]]] = []
    queue_ind = 0
    while all_needed_workers > 0:
        time, workers = time_resources_queue[queue_ind]
        next_workers: Dict[str, Worker] = {}
        for worker in workers.values():
            worker = worker.copy()
            used_count = 0
            if worker.name in needed_workers and needed_workers[worker.name] > 0:
                start_time = max(time, start_time)
                used_count = min(needed_workers[worker.name], worker.count)
                needed_workers[worker.name] -= used_count
                all_needed_workers -= used_count
            if worker.count - used_count > 0:
                worker.count -= used_count
                next_workers[worker.contractor_id] = worker
        next_time_resources_queue.append((time, next_workers))
        queue_ind += 1
    next_time_resources_queue.extend(time_resources_queue[queue_ind:])
    return start_time, next_time_resources_queue


def find_finish_time(start_time: Time, index: int, index2node: Dict[int, GraphNode], work_id2index: Dict[str, int],
                     worker_team: List[Worker], scheduled_work: Dict[str, ScheduledWork],
                     work_estimator: WorkTimeEstimator = None) -> Time:
    node = index2node[index]
    finish_time = start_time + node.work_unit.estimate_static(worker_team, work_estimator)
    work_id = node.id
    scheduled_work[work_id] = init_scheduled_work(start_time, finish_time, worker_team, node.work_unit)
    while node.inseparable_parent:
        work_id = node.inseparable_son.id
        node = index2node[work_id2index[work_id]]
        start_time = finish_time
        finish_time = start_time + node.work_unit.estimate_static(worker_team, work_estimator)
        scheduled_work[work_id] = init_scheduled_work(start_time, finish_time, worker_team, node.work_unit)
    return finish_time


def init_scheduled_work(start_time: Time, finish_time: Time, worker_team: List[Worker], work_unit: WorkUnit):
    return ScheduledWork(start_end_time=(start_time, finish_time),
                         workers=worker_team,
                         work_unit=work_unit)
