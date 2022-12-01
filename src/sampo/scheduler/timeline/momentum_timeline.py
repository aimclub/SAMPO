from collections import deque
from typing import Dict, List, Tuple, Optional, Iterable, Set, Union

from sortedcontainers import SortedList

from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.types import AgentId, ScheduleEvent, EventType
from sampo.utilities.collections import build_index


class MomentumTimeline(Timeline):

    def __init__(self, tasks: List[GraphNode], contractors: List[Contractor]):
        """
            This should create empty Timeline from given list of tasks and contractor list
            :param tasks:
            :param contractors:
            :return:
            """

        # using  time, seq_id and event_type we can guarantee that
        # there may be only one possible order in cases:
        # (a) when events have the same time
        # (in this cases we need both time and seq_id to properly handle
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
        self._timeline = {
            c.id: {
                w_name: SortedList(
                    iterable=(ScheduleEvent(-1, EventType.Initial, Time(0), None, ws.count),),
                    key=event_cmp
                )
                for w_name, ws in c.workers.items()
            }
            for c in contractors
        }

    def find_min_start_time(self, node: GraphNode, worker_team: List[Worker],
                            node2swork: Dict[GraphNode, ScheduledWork],
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        """
        Computes start time, max parent time, contractor and exec times for given node
        :param worker_team: list of passed workers. Should be IN THE SAME ORDER AS THE CORRESPONDING WREQS
        :param node:
        :param node2swork:
        :param work_estimator:
        :return:
        """
        return self.find_min_start_time_with_additional(node, worker_team, node2swork, work_estimator)[0]

    def find_min_start_time_with_additional(self, node: GraphNode, worker_team: List[Worker],
                                            contractor_id: str,
                                            node2swork: Dict[GraphNode, ScheduledWork],
                                            work_estimator: Optional[WorkTimeEstimator] = None) \
            -> Tuple[Time, Time, Dict[GraphNode, Tuple[Time, Time]]]:
        """
        Computes start time, max parent time, contractor and exec times for given node
        :param worker_team: list of passed workers. Should be IN THE SAME ORDER AS THE CORRESPONDING WREQS
        :param node:
        :param node2swork:
        :param work_estimator:
        :return:
        """
        inseparable_chain = node.get_inseparable_chain_with_self()
        # 1. identify earliest possible start time by max parent's end time

        max_parent_time: Time = max((node2swork[pnode].finish_time for pnode in node.parents), default=Time(0))
        nodes_start_times: Dict[GraphNode, Time] = {n: max((node2swork[pnode].finish_time
                                                            if pnode in node2swork else Time(0)
                                                            for pnode in n.parents),
                                                           default=Time(0))
                                                    for n in inseparable_chain}

        # 2. calculating execution time of the task

        exec_time: Time = Time(0)
        exec_times: Dict[GraphNode, Tuple[Time, Time]] = {}  # node: (lag, exec_time)
        for chain_node in inseparable_chain:
            passed_agents_new = [agent.copy() for agent in worker_team]

            node_exec_time: Time = Time(0) if len(chain_node.work_unit.worker_reqs) == 0 else \
                chain_node.work_unit.estimate_static(passed_agents_new, work_estimator)
            lag_req = nodes_start_times[chain_node] - max_parent_time - exec_time
            lag = lag_req if lag_req > 0 else 0

            exec_times[chain_node] = lag, node_exec_time
            exec_time += lag + node_exec_time

        if len(worker_team) == 0:
            return max_parent_time, max_parent_time, exec_times

        return self._find_min_start_time(self._timeline[contractor_id], inseparable_chain, max_parent_time,
                                         exec_time, worker_team), max_parent_time, exec_times

    def _find_min_start_time(self,
                             resource_timeline: Dict[str, SortedList[ScheduleEvent]],
                             inseparable_chain: List[GraphNode],
                             parent_time: Time,
                             exec_time: Time,
                             passed_agents: List[Worker]) -> Time:
        # if it is a service unit, than it can be satisfied by any contractor at any moment
        # because no real workers is going to be used to execute the task
        # however, we still should respect dependencies of the service task
        # and should start it only after all the dependencies tasks are done
        if all((node.work_unit.is_service_unit for node in inseparable_chain)):
            return parent_time

        # checking if the contractor can satisfy requirements for the task at all
        # we return None in cases when the task cannot be executed
        # even if it is scheduled to the very end, e.g. after the end of all other tasks
        # already scheduled to this contractor

        for node in inseparable_chain:
            for i, wreq in enumerate(node.work_unit.worker_reqs):
                initial_event: ScheduleEvent = resource_timeline[wreq.kind][0]
                assert initial_event.event_type is EventType.Initial
                # if this contractor initially has fewer workers of this type, then needed...
                if initial_event.available_workers_count < passed_agents[i].count:
                    return Time.inf()

        # here we look for the earliest time slot that can satisfy all the worker's specializations
        # we do it in that manner because each worker specialization can be treated separately
        # e.g. requested for different tasks
        # We check only the first node since all inseparable nodes have same worker_reqs despite the difference in exec time
        queue = deque(inseparable_chain[0].work_unit.worker_reqs)

        start = parent_time
        scheduled_wreqs: List[WorkerReq] = []

        type2count: Dict[str, int] = build_index(passed_agents, lambda w: w.name, lambda w: w.count)

        i = 0
        while len(queue) > 0:
            if i > 0 and i % 50 == 0:
                print(f"Warning! Probably cycle in looking for diff workers: {i} iteration")
            i += 1

            wreq = queue.popleft()
            state = resource_timeline[wreq.kind]
            # we look for the earliest time slot starting from 'start' time moment
            # if we have found a time slot for the previous task,
            # we should start to find for the earliest time slot of other task since this new time
            found_start = self._find_earliest_time_slot(state, start, exec_time, type2count[wreq.kind])

            assert found_start >= start

            if len(scheduled_wreqs) == 0 or start == found_start:
                # we schedule the first worker's specialization or the next spec has the same start time
                # as the all previous ones
                scheduled_wreqs.append(wreq)
                start = max(found_start, start)
            else:
                # The current worker specialization can be started only later than
                # the previously found start time.
                # In this case we need to add back all previously scheduled wreq-s into the queue
                # to be scheduled again with the new start time (e.g. found start).
                # This process should reach its termination at least at the very end of this contractor's schedule.
                queue.extend(scheduled_wreqs)
                scheduled_wreqs.clear()
                scheduled_wreqs.append(wreq)
                start = max(found_start, start)

        return start

    def _find_earliest_time_slot(self,
                                 state: SortedList[ScheduleEvent],
                                 parent_time: Time,
                                 exec_time: Time,
                                 required_worker_count: int) -> Time:
        current_start_time = parent_time
        current_start_idx = state.bisect_right(current_start_time) - 1

        # the condition means we have reached the end of schedule for this contractor subject to specialization (wreq)
        # as long as we assured that this contractor has enough capacity at all to handle the the task
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
            # we need to check the event current_start_idx - 1 cause it is the first event
            # that influence amount of available for us workers
            not_enough_workers_found = False
            for idx in range(end_idx - 1, current_start_idx - 2, -1):
                if state[idx].available_workers_count < required_worker_count or state[idx].time < parent_time:
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

    def update_timeline(self, finish: Time, worker_team: List[Worker]):
        """
        Adds given `worker_team` to the timeline at the moment `finish`
        :param finish:
        :param worker_team:
        :return:
        """
        # For each worker type consume the nearest available needed worker amount
        # and re-add it to the time when current work should be finished.
        # Addition performed as step in bubble-sort algorithm.
        for worker in worker_team:
            needed_count = worker.count
            worker_timeline = self._timeline[(worker.contractor_id, worker.name)]
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

    def schedule(self,
                 task_index: int,
                 node: GraphNode,
                 id2swork: Dict[GraphNode, ScheduledWork],
                 workers: List[Worker],
                 contractor: Contractor,
                 assigned_time: Optional[Time],
                 work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        inseparable_chain = node.get_inseparable_chain_with_self()
        st = self.find_min_start_time(node, workers, id2swork)
        if assigned_time:
            exec_times = {n: (Time(0), assigned_time // len(inseparable_chain))
                          for n in inseparable_chain}
            return self._schedule_with_inseparables(id2swork, workers, contractor, inseparable_chain, st, exec_times,
                                                    work_estimator)
        else:
            return self._schedule_with_inseparables(id2swork, workers, contractor, inseparable_chain, st, {},
                                                    work_estimator)

    def __getitem__(self, item: AgentId):
        return self._timeline[item]


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
