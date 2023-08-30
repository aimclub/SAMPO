from collections import deque
from dataclasses import dataclass
from operator import attrgetter

from sortedcontainers import SortedList

from sampo.schemas.sorted_list import ExtendedSortedList
from sampo.schemas.time import Time
from sampo.schemas.zones import ZoneReq, ZoneConfiguration, Zone
from sampo.utilities.collections_util import build_index


@dataclass
class ZoneScheduleEvent:
    time: Time
    status: int


class ZoneTimeline:

    def __init__(self, config: ZoneConfiguration):
        self._timeline = {zone: ExtendedSortedList([ZoneScheduleEvent(Time(0), status)], key=attrgetter('time'))
                          for zone, status in config.start_statuses.items()}
        self._config = config

    def find_min_start_time(self, zones: list[ZoneReq], parent_time: Time, exec_time: Time):
        # here we look for the earliest time slot that can satisfy all the zones

        start = parent_time
        scheduled_wreqs: list[ZoneReq] = []

        type2count: dict[str, int] = build_index(zones, lambda w: w.name, lambda w: w.required_status)

        queue = deque(zones)

        i = 0
        while len(queue) > 0:
            i += 1

            wreq = queue.popleft()
            state = self._timeline[wreq.name]
            # we look for the earliest time slot starting from 'start' time moment
            # if we have found a time slot for the previous task,
            # we should start to find for the earliest time slot of other task since this new time
            found_start = self._find_earliest_time_slot(state, start, exec_time, type2count[wreq.name])

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
                                 state: SortedList[ZoneScheduleEvent],
                                 parent_time: Time,
                                 exec_time: Time,
                                 required_status: int) -> Time:
        """
        Searches for the earliest time starting from start_time, when a time slot
        of exec_time is available, when required_worker_count of resources is available

        :param state: stores Timeline for the certain resource
        :param parent_time: the minimum start time starting from the end of the parent task
        :param exec_time: execution time of work
        :param required_status: requirements status of zone
        :return: the earliest start time
        """
        current_start_time = parent_time
        current_start_idx = state.bisect_right(current_start_time) - 1

        # the condition means we have reached the end of schedule for this contractor subject to specialization (wreq)
        # as long as we assured that this contractor has enough capacity at all to handle the task
        # we can stop and put the task at the very end
        i = 0
        while len(state[current_start_idx:]) > 0:
            # if i > 0 and i % 50 == 0:
            #     print(f'Warning! Probably cycle in looking for earliest time slot: {i} iteration')
            #     print(f'Current start time: {current_start_time}, current start idx: {current_start_idx}')
            i += 1
            end_idx = state.bisect_right(current_start_time + exec_time)

            # checking from the end of execution interval, i.e., end_idx - 1
            # up to (including) the event right prepending the start
            # of the execution interval, i.e., current_start_idx - 1
            # we need to check the event current_start_idx - 1 cause it is the first event
            # that influence amount of available for us workers
            not_compatible_status_found = False
            for idx in range(end_idx - 1, current_start_idx - 2, -1):
                if not self._config.statuses.match_status(required_status, state[idx].status) \
                        or state[idx].time < parent_time:
                    # we're trying to find a new slot that would start with
                    # either the last index passing the quantity check
                    # or the index after the execution interval
                    # we need max here to process a corner case when the problem arises
                    # on current_start_idx - 1
                    # without max it would get into infinite cycle
                    current_start_idx = max(idx, current_start_idx) + 1
                    not_compatible_status_found = True
                    break

            if not not_compatible_status_found:
                break

            if current_start_idx >= len(state):
                break

            current_start_time = state[current_start_idx].time

        return current_start_time

    def update_timeline(self, zones: list[Zone], start_time: Time, exec_time: Time):
        for zone in zones:
            state = self._timeline[zone.name]
            start_idx = state.bisect_right(start_time)
            end_idx = state.bisect_right(start_time + exec_time)
            start_status = state[start_idx - 1].status
            # updating all events in between the start and the end of our current task
            for event in state[start_idx: end_idx]:
                # TODO Check that we shouldn't change the between statuses
                assert self._config.statuses.match_status(zone.status, event[1])
                # event.available_workers_count -= w.count

            assert self._config.statuses.match_status(zone.status, start_status)

            state.add(ZoneScheduleEvent(start_time, zone.status))
