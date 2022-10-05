import logging
from enum import Enum
from uuid import uuid4
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Union, Iterable

import numpy as np
from sortedcontainers import SortedList

from schemas.work_estimator import WorkTimeEstimator
from schemas.schedule import ScheduledWork
from schemas.contractor import Contractor, get_agents_from_contractors, AgentsDict
from schemas.requirements import WorkerReq
from schemas.resources import Worker
from schemas.time import Time
from schemas.graph import GraphNode, WorkGraph

logger = logging.getLogger(__name__)


class EventType(Enum):
    Initial = -1
    End = 0
    Start = 1

    @property
    def priority(self):
        return self.value


@dataclass
class ScheduleEvent:
    seq_id: int
    event_type: EventType
    time: Time
    swork: Optional[ScheduledWork]
    available_workers_count: int


def build_schedule_from_task_sequence(tasks: List[GraphNode], _: WorkGraph, contractors: List[Contractor],
                                      work_estimator: WorkTimeEstimator = None)\
        -> Iterable[ScheduledWork]:
    """
    Builds a schedule from a list of tasks where all dependents tasks are guaranteed to be later
    in the sequence than their dependencies
    :param work_estimator:
    :param tasks: list of tasks ordered by some algorithm according to their dependencies and priorities
    :param _: graph of tasks to be executed
    :param contractors: pools of workers available for execution
    :return: a schedule
    """
    # now we may work only with workers that have
    # only workers with the same productivity
    # (e.g. for each specialization each contractor has only one worker object)
    # check_all_workers_have_same_qualification(wg, contractors)

    # data structure to hold scheduled tasks
    node2swork: Dict[GraphNode, ScheduledWork] = dict()

    # using  time, seq_id and event_type we can guarantee that
    # there may be only one possible order in cases:
    # (a) when events have the same time
    # (in these cases we need both time and seq_id to properly handle
    # available_workers processing logic)
    # (b) when events have the same time and their start and end matches
    # (service tasks for instance may have zero length)
    def event_cmp(x: Union[ScheduleEvent, Time, Tuple[Time, int, int]]) -> Tuple[Time, int, int]:
        if isinstance(x, ScheduleEvent):
            if x.event_type is EventType.Initial:
                return Time(-1), -1, x.event_type.priority

            return x.time, x.seq_id, x.event_type.priority

        if isinstance(x, Time):
            return x, len(tasks), 1

        if isinstance(x, tuple):
            return x

        raise ValueError(f"Incorrect type of value: {type(x)}")

    # to efficiently search for time slots for tasks to be scheduled
    # we need to keep track of starts and ends of previously scheduled tasks
    # and remember how many workers of a certain type is available at this particular moment
    contractors_state: Dict[str, Dict[str, SortedList[ScheduleEvent]]] = {
        c.id: {
            w_name: SortedList(
                iterable=(ScheduleEvent(-1, EventType.Initial, Time(0), None, sum(w.count for w in ws)),),
                key=event_cmp
            )
            for w_name, ws in c.worker_types.items()
        }
        for c in contractors
    }

    # we can get agents here, because they are always same and not updated
    agents = get_agents_from_contractors(contractors)

    # We allocate resources for the whole inseparable chain, when we process the first node in it.
    # So, we will store IDs of non-head nodes in such chains to skip them.
    # Note that tasks are already topologically ordered,
    # i.e., the first node in a chain is always processed before its children

    skipped_inseparable_children: Set[str] = set()
    # scheduling all the tasks in a one-by-one manner
    for i, node in enumerate(tasks):
        # skip, if this node was processed as a part of an inseparable chin previously
        if node.id in skipped_inseparable_children:
            continue

        # 0. find, if the node starts an inseparable chain

        inseparable_chain = node.get_inseparable_chain()
        if inseparable_chain:
            skipped_inseparable_children.update((ch.id for ch in inseparable_chain))
        whole_work_nodes = inseparable_chain if inseparable_chain else [node]

        # 1. identify the earliest possible start time by max parent's end time

        start_time: Time = max((node2swork[p_node].finish_time for p_node in node.parent_nodes), default=Time(0))
        nodes_start_times: Dict[GraphNode, Time] = {n: max((node2swork[p_node].finish_time
                                                            if p_node in node2swork else Time(0)
                                                            for p_node in n.parent_nodes),
                                                           default=Time(0))
                                                    for n in whole_work_nodes}

        # 2. calculating execution time of the task

        exec_time: Time = Time(0)
        exec_times: Dict[GraphNode, Tuple[Time, Time]] = {}  # node: (lag, exec_time)
        for chain_node in whole_work_nodes:
            passed_agents = []
            for req in chain_node.work_unit.worker_reqs:
                a = Worker(str(uuid4()), req.type,
                           get_worker_count(req.min_count, req.max_count,
                                            max(list(map(lambda x: x.count, agents[req.type].values())))))
                passed_agents.append(a)

            node_exec_time: Time = Time(0) if len(chain_node.work_unit.worker_reqs) == 0 else \
                chain_node.work_unit.estimate_static(passed_agents, work_estimator)
            lag_req = nodes_start_times[chain_node] - start_time - exec_time
            lag = lag_req if lag_req > 0 else 0

            exec_times[chain_node] = lag, node_exec_time
            exec_time += lag + node_exec_time

        # 3. for each contractor we find the earliest possible time slot
        # that can satisfy required amount of workers for all specializations
        # for the planned duration of the task (e.g. execution time)

        start_times = list(
            _find_time_slot_for_all_workers(
                contractors_state[c.id],
                whole_work_nodes,
                start_time,
                exec_time,
                agents
            ) for c in contractors
        )
        # None in the list of contractors means this contractor cannot satisfy the task at all
        # even if it is scheduled to the very end, e.g.after the end of all other tasks
        # already scheduled to this contractor
        # there should be at least 1 non-None contractor
        assert any(st is not None for st in start_times)

        # 4. We choose a certain contractor to execute the task
        # (up to now it is the contractor with the earliest possible start time)

        min_idx = np.nanargmin([t if t is not None else np.nan for t in start_times])
        min_idx = min_idx if isinstance(min_idx, np.ndarray) else min_idx
        start_time: Time = start_times[min_idx]
        contractor = contractors[min_idx]

        # 5. form a set of workers objects to be assigned to the task

        chosen_workers = [
            Worker("", w_req.type,
                   contractor_id=contractor.id,
                   productivity_class=contractor.worker_types[w_req.type][0].productivity_class,
                   productivity=contractor.worker_types[w_req.type][0].productivity,
                   count=get_worker_count(w_req.min_count,
                                          max(list(map(lambda x: x.count, agents[w_req.type].values()))),
                                          w_req.max_count))
            for w_req in node.work_unit.worker_reqs
        ]

        # 6. create a schedule entry for the task

        curr_time = start_time
        for chain_node in whole_work_nodes:
            node_lag, node_time = exec_times[chain_node]
            start_work = curr_time + node_lag
            swork = ScheduledWork(
                work_unit=chain_node.work_unit,
                start_end_time=(start_work, start_work + node_time),
                workers=chosen_workers
            )
            curr_time += node_time + node_lag
            node2swork[chain_node] = swork

        # 7. for each worker's specialization of the chosen contractor being used by the task
        # we update counts of available workers on previously scheduled events
        # that lie between start and end of the task
        # Also, add events of the start and the end to worker's specializations
        # of the chosen contractor.

        # experimental logics lightening. debugging showed its efficiency.

        swork = node2swork[node]  # masking the whole chain ScheduleEvent with the first node
        start = swork.start_time
        end = node2swork[whole_work_nodes[-1]].finish_time
        for w in chosen_workers:
            state = contractors_state[w.contractor_id][w.name]
            start_idx = state.bisect_right(start)
            end_idx = state.bisect_right(end)
            available_workers_count = state[start_idx - 1].available_workers_count
            # updating all events in between the start and the end of our current task
            for event in state[start_idx: end_idx]:
                # assert event.available_workers_count >= w.count
                event.available_workers_count -= w.count

            # assert available_workers_count >= w.count

            if start_idx < end_idx:
                event: ScheduleEvent = state[end_idx - 1]
                # assert state[0].available_workers_count >= event.available_workers_count + w.count
                end_count = event.available_workers_count + w.count
            else:
                # assert state[0].available_workers_count >= available_workers_count
                end_count = available_workers_count

            state.add(ScheduleEvent(i, EventType.Start, start, swork, available_workers_count - w.count))
            state.add(ScheduleEvent(i, EventType.End, end, swork, end_count))

    # 8. form the final Schedule object
    # schedule = Schedule(workGraph=wg, resourcePools=deepcopy(contractors), works=list(node2swork.values()))
    # return schedule

    return node2swork.values()


def _find_time_slot_for_all_workers(
        cont_state: Dict[str, SortedList[ScheduleEvent]],
        nodes: List[GraphNode],
        start_time: Time,
        exec_time: Time,
        agents: AgentsDict) -> Optional[Time]:
    # if it is a service unit, then it can be satisfied by any contractor at any moment
    # because no real workers is going to be used to execute the task
    # however, we still should respect dependencies of the service task
    # and should start it only after all the dependencies tasks are done
    if all((node.work_unit.is_service_unit for node in nodes)):
        return start_time

    # checking if the contractor can satisfy requirements for the task at all
    # we return None in cases when the task cannot be executed
    # even if it is scheduled to the very end, e.g. after the end of all other tasks
    # already scheduled to this contractor

    for node in nodes:
        for w_req in node.work_unit.worker_reqs:
            initial_event: ScheduleEvent = cont_state[w_req.type][0]
            assert initial_event.event_type is EventType.Initial
            # if this contractor initially has fewer workers of this type, then needed...
            if initial_event.available_workers_count < (
                    get_worker_count(w_req.min_count,
                                     max(list(map(lambda x: x.count, agents[w_req.type].values()))),
                                     w_req.max_count)):
                return None

    # here we look for the earliest time slot that can satisfy all the worker's specializations
    # we do it in that manner because each worker specialization can be treated separately
    # e.g. requested for different tasks
    # We check only the first node since all inseparable nodes have same worker_reqs despite the difference in exec time
    queue = deque(nodes[0].work_unit.worker_reqs)

    start = start_time
    scheduled_w_reqs: List[WorkerReq] = []

    i = 0
    while len(queue) > 0:
        if i > 0 and i % 50 == 0:
            print(f"Warning! Probably cycle in looking for diff workers: {i} iteration")
        i += 1

        w_req = queue.popleft()
        state = cont_state[w_req.type]
        # we look for the earliest time slot starting from 'start' time moment
        # if we have found a time slot for the previous task,
        # we should start to find for the earliest time slot of other task since this new time
        found_start = _find_earliest_time_slot(state, w_req, start, exec_time, agents)

        assert found_start >= start

        if len(scheduled_w_reqs) == 0 or start == found_start:
            # we schedule the first worker's specialization or the next spec has the same start time
            # as the all previous ones
            scheduled_w_reqs.append(w_req)
            start = max(found_start, start)
        else:
            # The current worker specialization can be started only later than
            # the previously found start time.
            # In this case we need to add back all previously scheduled w_req-s into the queue
            # to be scheduled again with the new start time (e.g. found start).
            # This process should reach its termination at least at the very end of this contractor's schedule.
            queue.extend(scheduled_w_reqs)
            scheduled_w_reqs.clear()
            scheduled_w_reqs.append(w_req)
            start = max(found_start, start)

    return start


def _find_earliest_time_slot(
        state: SortedList[ScheduleEvent],
        w_req: WorkerReq,
        start_time: Time,
        exec_time: Time,
        agents: AgentsDict) -> Time:
    required_worker_count = get_worker_count(w_req.min_count,
                                             max(list(map(lambda x: x.count, agents[w_req.type].values()))),
                                             w_req.max_count)
    current_start_time = start_time
    current_start_idx = state.bisect_right(current_start_time) - 1

    # the condition means we have reached the end of schedule for this contractor subject to specialization (w_req)
    # as long as we assured that this contractor has enough capacity at all to handle the task
    # we can stop and put the task at the very end
    i = 0
    while len(state[current_start_idx:]) > 0:
        if i > 0 and i % 50 == 0:
            print(f"Warning! Probably cycle in looking for earliest time slot: {i} iteration")
            print(f"Current start time: {current_start_time}, current start idx: {current_start_idx}")
        i += 1
        end_idx = state.bisect_right(current_start_time + exec_time)

        # checking from the end of execution interval, i.e., end_idx - 1
        # up to (including) the event right prepending the start of the execution interval, i.e., current_start_idx - 1
        # we need to check the event current_start_idx - 1 because it is the first event
        # that influence amount of available for us workers
        not_enough_workers_found = False
        for idx in range(end_idx - 1, current_start_idx - 2, -1):
            if state[idx].available_workers_count < required_worker_count or state[idx].time < start_time:
                # we're trying to find a new slot that would start with
                # either the last index passing the quantity check
                # or the index after the execution interval
                # we need max here to process a corner case when the problem arises
                # on current_start_idx - 1
                # without max it would get into infinite cycle
                current_start_idx = max(idx, current_start_idx) + 1
                not_enough_workers_found = True
                break

        if not not_enough_workers_found:
            break

        if current_start_idx >= len(state):
            break

        current_start_time = state[current_start_idx].time

    return current_start_time


def get_worker_count(min_req: int, max_req: int, available: int):
    return (min_req + min(available, max_req)) // 2
    # return min_req
