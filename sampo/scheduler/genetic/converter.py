from typing import List, Tuple

import numpy as np

from sampo.scheduler.base import Scheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator

ChromosomeType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def convert_schedule_to_chromosome(wg: WorkGraph,
                                   work_id2index: dict[str, int], worker_name2index: dict[str, int],
                                   contractor2index: dict[str, int], contractor_borders: np.ndarray,
                                   schedule: Schedule, order: list[GraphNode] | None = None) -> ChromosomeType:
    """
    Receive result of scheduling algorithm and transform it to chromosome

    :param wg:
    :param work_id2index:
    :param worker_name2index:
    :param contractor2index:
    :param contractor_borders:
    :param schedule:
    :param order: if passed, specifies the node order that should appear in the chromosome
    :return:
    """

    order: list[GraphNode] = order if order is not None else [work for work in schedule.works
                                                              if not wg[work.work_unit.id].is_inseparable_son()]

    # order works part of chromosome
    order_chromosome: np.ndarray = np.array([work_id2index[work.work_unit.id] for work in order])

    # convert to convenient form
    schedule = schedule.to_schedule_work_dict

    # resources for works part of chromosome
    # +1 stores contractors line
    resource_chromosome = np.zeros((len(worker_name2index) + 1, len(order_chromosome)), dtype=int)

    for node in order:
        node_id = node.work_unit.id
        index = work_id2index[node_id]
        for resource in schedule[node_id].workers:
            res_count = resource.count
            res_index = worker_name2index[resource.name]
            res_contractor = resource.contractor_id
            resource_chromosome[res_index, index] = res_count
            resource_chromosome[-1, index] = contractor2index[res_contractor]

    resource_border_chromosome = np.copy(contractor_borders)

    return order_chromosome, resource_chromosome, resource_border_chromosome


def convert_chromosome_to_schedule(chromosome: ChromosomeType, worker_pool: WorkerContractorPool,
                                   index2node: dict[int, GraphNode],
                                   index2contractor: dict[int, Contractor],
                                   worker_pool_indices: dict[int, dict[int, Worker]],
                                   spec: ScheduleSpec,
                                   timeline: Timeline | None = None,
                                   assigned_parent_time: Time = Time(0),
                                   work_estimator: WorkTimeEstimator = None, ) \
        -> tuple[dict[GraphNode, ScheduledWork], Time, Timeline]:
    node2swork: dict[GraphNode, ScheduledWork] = {}

    if not isinstance(timeline, JustInTimeTimeline):
        timeline = JustInTimeTimeline(index2node.values(), index2contractor.values(), worker_pool)
    works_order = chromosome[0]
    works_resources = chromosome[1]

    for order_index, work_index in enumerate(works_order):
        node = index2node[work_index]
        # if node.id in node2swork and not node.is_inseparable_son():
        #     continue

        work_spec = spec.get_work_spec(node.id)

        resources = works_resources[:-1, work_index]
        contractor_index = works_resources[-1, work_index]
        contractor = index2contractor[contractor_index]
        worker_team: List[Worker] = [worker_pool_indices[worker_index][contractor_index]
                                     .copy().with_count(worker_count)
                                     for worker_index, worker_count in enumerate(resources)
                                     if worker_count > 0]

        # apply worker spec
        Scheduler.optimize_resources_using_spec(node.work_unit, worker_team, work_spec)

        st = timeline.find_min_start_time(node, worker_team, node2swork, assigned_parent_time, work_estimator)

        if order_index == 0:  # we are scheduling the work `start of the project`
            st = assigned_parent_time  # this work should always have st = 0, so we just re-assign it

        # finish using time spec
        timeline.schedule(order_index, node, node2swork, worker_team, contractor,
                          st, work_spec.assigned_time, assigned_parent_time, work_estimator)

    schedule_start_time = min([swork.start_time for swork in node2swork.values() if
                               len(swork.work_unit.worker_reqs) != 0])

    return node2swork, schedule_start_time, timeline
