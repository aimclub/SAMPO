from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np

from sampo.scheduler.base import Scheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit

ChromosomeType = Tuple[List[int], np.ndarray]


def convert_schedule_to_chromosome(index2node: Dict[int, GraphNode],
                                   work_id2index: Dict[str, int], worker_name2index: Dict[str, int],
                                   contractor2index: Dict[str, int],
                                   schedule: Schedule) -> Tuple[List[int], np.ndarray]:
    """
    received result of scheduling algorithm and transform it to chromosome
    :param contractor2index:
    :param work_id2index:
    :param index2node:
    :param worker_name2index:
    :param schedule:
    :return:
    """

    # order works part of chromosome
    order_chromosome: List[int] = [work_id2index[swork.work_unit.id] for swork in schedule.works]

    # convert to convenient form
    schedule = schedule.to_schedule_work_dict

    # resources for works part of chromosome
    # +1 stores contractors line
    resource_chromosome = np.zeros((len(worker_name2index) + 1, len(order_chromosome)), dtype=int)

    for index, node in index2node.items():
        node_reqs = set([req.kind for req in node.work_unit.worker_reqs])
        for resource in schedule[node.id].workers:
            if resource.name in node_reqs:
                res_count = resource.count
                res_index = worker_name2index[resource.name]
                res_contractor = resource.contractor_id
                work_index = work_id2index[node.id]
                resource_chromosome[res_index, work_index] = res_count
                resource_chromosome[-1, work_index] = contractor2index[res_contractor]

    return order_chromosome, resource_chromosome


def convert_chromosome_to_schedule(chromosome: ChromosomeType, worker_pool: WorkerContractorPool,
                                   index2node: Dict[int, GraphNode],
                                   worker_name2index: Dict[str, int],
                                   index2contractor: Dict[int, Contractor],
                                   spec: ScheduleSpec,
                                   work_estimator: WorkTimeEstimator = None) \
        -> tuple[Dict[GraphNode, ScheduledWork], Timeline]:
    node2swork: Dict[GraphNode, ScheduledWork] = {}

    timeline = JustInTimeTimeline(worker_pool)
    works_order = chromosome[0]
    works_resources = chromosome[1]
    for order_index, work_index in enumerate(works_order):
        node = index2node[work_index]
        if node.id in node2swork and not node.is_inseparable_son():
            continue

        work_spec = spec.get_work_spec(node.id)

        resources = works_resources[:-1, work_index]
        contractor = index2contractor[works_resources[-1, work_index]]
        worker_team: List[Worker] = [Worker(str(uuid4()), worker_name, resources[worker_index],
                                            contractor_id=contractor.id)
                                     for worker_name, worker_index in worker_name2index.items()
                                     if resources[worker_index] > 0]

        # apply worker spec
        Scheduler.optimize_resources_using_spec(node.work_unit, worker_team, work_spec)

        # finish using time spec
        finish_time = timeline.schedule(order_index, node, node2swork, worker_team, contractor,
                                        None, work_spec.assigned_time, work_estimator)

        timeline.update_timeline(order_index, finish_time, node, node2swork, worker_team)
    return node2swork, timeline


def init_scheduled_work(start_time: Time, finish_time: Time, worker_team: List[Worker],
                        contractor: Contractor, work_unit: WorkUnit):
    return ScheduledWork(start_end_time=(start_time, finish_time),
                         workers=worker_team,
                         work_unit=work_unit,
                         contractor=contractor)
