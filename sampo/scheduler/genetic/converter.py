import copy

import numpy as np

from sampo.scheduler.base import Scheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.general_timeline import GeneralTimeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.requirements import ZoneReq
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.linked_list import LinkedList

ChromosomeType = tuple[np.ndarray, np.ndarray, np.ndarray, ScheduleSpec, np.ndarray]


def convert_schedule_to_chromosome(work_id2index: dict[str, int],
                                   worker_name2index: dict[str, int],
                                   contractor2index: dict[str, int],
                                   contractor_borders: np.ndarray,
                                   schedule: Schedule,
                                   spec: ScheduleSpec,
                                   landscape: LandscapeConfiguration,
                                   order: list[GraphNode] | None = None) -> ChromosomeType:
    """
    Receive a result of scheduling algorithm and transform it to chromosome

    :param work_id2index:
    :param worker_name2index:
    :param contractor2index:
    :param contractor_borders:
    :param schedule:
    :param spec:
    :param landscape:
    :param order: if passed, specify the node order that should appear in the chromosome
    :return:
    """

    order: list[ScheduledWork] = order if order is not None else [work for work in schedule.works
                                                                  if work.id in work_id2index]

    # order works part of chromosome
    order_chromosome: np.ndarray = np.array([work_id2index[work.id] for work in order])

    # convert to convenient form
    schedule = schedule.to_schedule_work_dict

    # resources for works part of chromosome
    # +1 stores contractors line
    resource_chromosome = np.zeros((len(order_chromosome), len(worker_name2index) + 1), dtype=int)

    # zone status changes after node executing
    zone_changes_chromosome = np.zeros((len(order_chromosome), len(landscape.zone_config.start_statuses)), dtype=int)

    for node in order:
        node_id = node.id
        index = work_id2index[node_id]
        for resource in schedule[node_id].workers:
            res_count = resource.count
            res_index = worker_name2index[resource.name]
            res_contractor = resource.contractor_id
            resource_chromosome[index, res_index] = res_count
            resource_chromosome[index, -1] = contractor2index[res_contractor]

    resource_border_chromosome = np.copy(contractor_borders)

    return order_chromosome, resource_chromosome, resource_border_chromosome, spec, zone_changes_chromosome


def convert_chromosome_to_schedule(chromosome: ChromosomeType,
                                   worker_pool: WorkerContractorPool,
                                   index2node: dict[int, GraphNode],
                                   index2contractor: dict[int, Contractor],
                                   index2zone: dict[int, str],
                                   worker_pool_indices: dict[int, dict[int, Worker]],
                                   worker_name2index: dict[str, int],
                                   contractor2index: dict[str, int],
                                   landscape: LandscapeConfiguration = LandscapeConfiguration(),
                                   timeline: Timeline | None = None,
                                   assigned_parent_time: Time = Time(0),
                                   work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
        -> tuple[dict[GraphNode, ScheduledWork], Time, Timeline, list[GraphNode]]:
    """
    Build schedule from received chromosome
    It can be used in visualization of final solving of genetic algorithm

    Here are Parallel SGS
    """
    node2swork: dict[GraphNode, ScheduledWork] = {}

    works_order = chromosome[0]
    works_resources = chromosome[1]
    border = chromosome[2]
    spec = chromosome[3]
    zone_statuses = chromosome[4]
    worker_pool = copy.deepcopy(worker_pool)

    # use 3rd part of chromosome in schedule generator
    for worker_index in worker_pool:
        for contractor_index in worker_pool[worker_index]:
            worker_pool[worker_index][contractor_index].with_count(border[contractor2index[contractor_index],
            worker_name2index[worker_index]])

    if not isinstance(timeline, JustInTimeTimeline):
        timeline = JustInTimeTimeline(worker_pool, landscape)

    order_nodes = []

    # timeline to store starts and ends of all works
    work_timeline = GeneralTimeline()

    def decode(work_index):
        cur_node = index2node[work_index]

        cur_work_spec = spec.get_work_spec(cur_node.id)
        cur_resources = works_resources[work_index, :-1]
        cur_contractor_index = works_resources[work_index, -1]
        cur_contractor = index2contractor[cur_contractor_index]
        cur_worker_team: list[Worker] = [worker_pool_indices[worker_index][cur_contractor_index]
                                         .copy().with_count(worker_count)
                                         for worker_index, worker_count in enumerate(cur_resources)
                                         if worker_count > 0]
        if cur_work_spec.assigned_time is not None:
            cur_exec_time = cur_work_spec.assigned_time
        else:
            cur_exec_time = work_estimator.estimate_time(cur_node.work_unit, cur_worker_team)
        return cur_node, cur_worker_team, cur_contractor, cur_exec_time, cur_work_spec

    # account the remaining works
    enumerated_works_remaining = LinkedList(iterable=enumerate(
        [(work_index, *decode(work_index)) for work_index in works_order]
    ))

    # declare current checkpoint index
    cpkt_idx = 0
    start_time = assigned_parent_time - 1

    def work_scheduled(args) -> bool:
        idx, (work_idx, node, worker_team, contractor, exec_time, work_spec) = args

        if timeline.can_schedule_at_the_moment(node, worker_team, work_spec, node2swork, start_time, exec_time):
            # apply worker spec
            Scheduler.optimize_resources_using_spec(node.work_unit, worker_team, work_spec)

            st = start_time
            if idx == 0:  # we are scheduling the work `start of the project`
                st = assigned_parent_time  # this work should always have st = 0, so we just re-assign it

            # finish using time spec
            ft = timeline.schedule(node, node2swork, worker_team, contractor, work_spec,
                                   st, exec_time, assigned_parent_time, work_estimator)

            work_timeline.update_timeline(st, exec_time, None)

            # process zones
            zone_reqs = [ZoneReq(index2zone[i], zone_status) for i, zone_status in enumerate(zone_statuses[work_idx])]
            zone_start_time = timeline.zone_timeline.find_min_start_time(zone_reqs, ft, 0)

            # we should deny scheduling
            # if zone status change can be scheduled only in delayed manner
            if zone_start_time != ft:
                node2swork[node].zones_post = timeline.zone_timeline.update_timeline(idx,
                                                                                     [z.to_zone() for z in zone_reqs],
                                                                                     zone_start_time, 0)
            return True
        return False

    # while there are unprocessed checkpoints
    while len(enumerated_works_remaining) > 0:
        if cpkt_idx < len(work_timeline):
            start_time = work_timeline[cpkt_idx]
            if start_time.is_inf():
                # break because schedule already contains Time.inf(), that is incorrect schedule
                break
        else:
            start_time += 1

        # find all works that can start at start_time moment
        enumerated_works_remaining.remove_if(work_scheduled)
        cpkt_idx = min(cpkt_idx + 1, len(work_timeline))

    return node2swork, assigned_parent_time, timeline, order_nodes
