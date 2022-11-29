from typing import Dict, List, Tuple, Optional, Iterable, Set

from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.scheduler.heft.time_computaion import calculate_working_time
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.types import AgentId

# stacks of time(Time) and count[int]
JustInTimeTimeline = Dict[AgentId, List[Tuple[Time, int]]]


def make_and_cache_schedule(id2swork: Dict[str, ScheduledWork],
                            workers: List[Worker],
                            contractor: Contractor,
                            inseparable_chain: List[GraphNode],
                            start_time: Time,
                            exec_times: Dict[GraphNode, Tuple[Time, Time]],
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
    """
    Makes ScheduledWork object from `GraphNode` and worker list, assigned `start_end_time`
    and adds it ti given `id2swork`. Also does the same for all inseparable nodes starts from this one
    :param node:
    :param id2swork:
    :param workers:
    :param contractor:
    :param inseparable_chain:
    :param start_time:
    :param exec_times:
    :param work_estimator:
    :return:
    """
    c_ft = start_time
    for dep_node in inseparable_chain:
        # set start time as finish time of original work
        # set finish time as finish time + working time of current node with identical resources
        # (the same as in original work)
        # set the same workers on it
        # TODO Decide where this should be
        max_parent_time = max((id2swork[pnode.id].finish_time
                               for pnode in dep_node.parents),
                              default=Time(0))
        working_time = exec_times.get(dep_node, None)
        if working_time is None:
            working_time = calculate_working_time(dep_node.work_unit, workers, work_estimator)
        new_finish_time = max_parent_time + working_time

        id2swork[dep_node.id] = ScheduledWork(work_unit=dep_node.work_unit,
                                              start_end_time=(max_parent_time, new_finish_time),
                                              workers=workers,
                                              contractor=contractor)
        # change finish time for using workers
        c_ft = new_finish_time

    return c_ft


def schedule_with_time_spec(node: GraphNode,
                            id2swork: Dict[str, ScheduledWork],
                            workers: List[Worker],
                            contractor: Contractor,
                            inseparable_chain: List[GraphNode],
                            timeline: JustInTimeTimeline,
                            assigned_time: Optional[Time],
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
    if assigned_time:
        st = find_min_start_time(node, workers, timeline, id2swork)
        exec_times = {n: (Time(0), assigned_time // len(inseparable_chain))
                      for n in inseparable_chain}
        return make_and_cache_schedule(id2swork, workers, contractor, inseparable_chain,
                                       st, exec_times, work_estimator)
    else:
        return schedule(node, id2swork, workers, contractor, inseparable_chain, timeline, work_estimator)


def find_min_start_time(node: GraphNode, worker_team: List[Worker],
                        timeline: JustInTimeTimeline,
                        id2swork: Dict[str, ScheduledWork]) -> Time:
    """
    Define the nearest possible start time for current job. It is equal the max value from:
    1. end time of all parent tasks
    2. time previous job off all needed workers to complete current task
    :param node: target node
    :param worker_team: worker team under testing
    :param timeline: save the end time of previous work for all workers
    :param id2swork:
    :return: found time, queue of employees without assigned workers
    """
    # if current job is the first
    if len(id2swork) == 0:
        return Time(0)
    # define the max end time of all parent tasks
    max_parent_time = max([id2swork[parent_node.id].finish_time
                           for parent_node in node.parents], default=Time(0))
    # define the max agents time when all needed workers are off from previous tasks
    max_agent_time = Time(0)

    # For each resource type
    for worker in worker_team:
        needed_count = worker.count
        offer_stack = timeline[worker.get_agent_id()]
        # Traverse list while not enough resources and grab it
        ind = len(offer_stack) - 1
        while needed_count > 0:
            offer_time, offer_count = offer_stack[ind]
            max_agent_time = max(max_agent_time, offer_time)

            if needed_count < offer_count:
                offer_count = needed_count
            needed_count -= offer_count
            ind -= 1

    return max(max_agent_time, max_parent_time)


def create_timeline(agents: WorkerContractorPool) -> JustInTimeTimeline:
    """
    Initializes basic resources timeline from agents dictionary. This structure should contain
    (sorted[descending] by free time) timeline for each type of worker defined as `AgentId`.
    Element of timeline have type `Tuple[Time, Int]`, so `Tuple[t, c]`
    means at the moment t there are c free workers of the corresponding `AgentId`
    :param agents:
    :return:
    """
    timeline: JustInTimeTimeline = {}
    for worker_type, worker_offers in agents.items():
        for worker_offer in worker_offers.values():
            timeline[worker_offer.get_agent_id()] = [(Time(0), worker_offer.count)]
    return timeline


def update_timeline(finish: Time, timeline: JustInTimeTimeline, worker_team: List[Worker]):
    """
    Adds given `worker_team` to the timeline at the moment `finish`
    :param finish:
    :param timeline:
    :param worker_team:
    :return:
    """
    # For each worker type consume the nearest available needed worker amount
    # and re-add it to the time when current work should be finished.
    # Addition performed as step in bubble-sort algorithm.
    for worker in worker_team:
        needed_count = worker.count
        worker_timeline = timeline[(worker.contractor_id, worker.name)]
        # Consume needed workers
        while needed_count > 0:
            next_time, next_count = worker_timeline.pop()
            if next_count > needed_count:
                worker_timeline.append((next_time, next_count - needed_count))
                break
            needed_count -= next_count

        # Add to the right place
        # worker_timeline.append((finish, worker.count))
        # worker_timeline.sort(reverse=True)
        worker_timeline.append((finish, worker.count))
        ind = len(worker_timeline) - 1
        while ind > 0 and worker_timeline[ind][0] > worker_timeline[ind - 1][0]:
            worker_timeline[ind], worker_timeline[ind - 1] = worker_timeline[ind - 1], worker_timeline[ind]
            ind -= 1
    return timeline


def schedule(node: GraphNode,
             id2swork: Dict[str, ScheduledWork],
             workers: List[Worker],
             contractor: Contractor,
             inseparable_chain: List[GraphNode],
             timeline: JustInTimeTimeline,
             work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
    """
    Makes ScheduledWork object from `GraphNode` and worker list, computes it's `start_end_time`
    and adds it ti given `id2swork`. Also does the same for all inseparable nodes starts from this one
    :param node:
    :param id2swork:
    :param workers:
    :param contractor:
    :param inseparable_chain:
    :param timeline:
    :param work_estimator:
    :return:
    """
    st = find_min_start_time(node, workers, timeline, id2swork)
    return make_and_cache_schedule(id2swork, workers, contractor, inseparable_chain, st, {}, work_estimator)


def order_nodes_by_start_time(works: Iterable[ScheduledWork], wg: WorkGraph) -> List[str]:
    """
    Makes ScheduledWorks' ordering that satisfies:
    1. Ascending order by start time
    2. Toposort
    :param works:
    :param wg:
    :return:
    """
    res = []
    order_by_start_time = [(item.start_time, item.work_unit.id) for item in
                           sorted(works, key=lambda item: item.start_time)]

    cur_time = 0
    cur_class: Set[GraphNode] = set()
    for start_time, work in order_by_start_time:
        node = wg[work]
        if len(cur_class) == 0:
            cur_time = start_time
        if start_time == cur_time:
            cur_class.add(node)
            continue
        # TODO Perform real toposort
        cur_not_added: Set[GraphNode] = set(cur_class)
        while len(cur_not_added) > 0:
            for cur_node in cur_class:
                if any([parent_node in cur_not_added for parent_node in cur_node.parents]):
                    continue  # we add this node later
                res.append(cur_node.id)
                cur_not_added.remove(cur_node)
            cur_class = set(cur_not_added)
        cur_time = start_time
        cur_class = {node}

    cur_not_added: Set[GraphNode] = set(cur_class)
    while len(cur_not_added) > 0:
        for cur_node in cur_class:
            if any([parent_node in cur_not_added for parent_node in cur_node.parents]):
                continue  # we add this node later
            res.append(cur_node.id)
            cur_not_added.remove(cur_node)
        cur_class = set(cur_not_added)

    return res
