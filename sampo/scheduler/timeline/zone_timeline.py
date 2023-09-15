from collections import deque

from sortedcontainers import SortedList

from sampo.schemas.requirements import ZoneReq
from sampo.schemas.time import Time
from sampo.schemas.types import EventType, ScheduleEvent
from sampo.schemas.zones import ZoneConfiguration, Zone, ZoneTransition
from sampo.utilities.collections_util import build_index


class ZoneTimeline:

    def __init__(self, config: ZoneConfiguration):
        def event_cmp(event: ScheduleEvent | Time | tuple[Time, int, int]) -> tuple[Time, int, int]:
            if isinstance(event, ScheduleEvent):
                if event.event_type is EventType.INITIAL:
                    return Time(-1), -1, event.event_type.priority

                return event.time, event.seq_id, event.event_type.priority

            if isinstance(event, Time):
                # instances of Time must be greater than almost all ScheduleEvents with same time point
                return event, Time.inf().value, 2

            if isinstance(event, tuple):
                return event

            raise ValueError(f'Incorrect type of value: {type(event)}')

        self._timeline = {zone: SortedList([ScheduleEvent(-1, EventType.INITIAL, Time(0), None, status)],
                                           key=event_cmp)
                          for zone, status in config.start_statuses.items()}
        self._config = config

    def find_min_start_time(self, zones: list[ZoneReq], parent_time: Time, exec_time: Time):
        # here we look for the earliest time slot that can satisfy all the zones

        start = parent_time
        scheduled_wreqs: list[ZoneReq] = []

        type2status: dict[str, int] = build_index(zones, lambda w: w.kind, lambda w: w.required_status)

        queue = deque(zones)

        i = 0
        while len(queue) > 0:
            # if i > 0 and i % 50 == 0:
            #     print(f'Warning! Probably cycle in looking for time slot for all reqs: {i} iteration')
            #     print(f'Current queue size: {len(queue)}')
            i += 1

            wreq = queue.popleft()
            state = self._timeline[wreq.kind]
            # we look for the earliest time slot starting from 'start' time moment
            # if we have found a time slot for the previous task,
            # we should start to find for the earliest time slot of other task since this new time
            found_start = self._find_earliest_time_slot(state, start, exec_time, type2status[wreq.kind])

            assert found_start >= start

            if len(scheduled_wreqs) == 0 or start == found_start:
                # we schedule the first worker's specialization or the next spec has the same start time
                # as the all previous ones
                scheduled_wreqs.append(wreq)
                start = max(start, found_start)
            else:
                # The current worker specialization can be started only later than
                # the previously found start time.
                # In this case we need to add back all previously scheduled wreq-s into the queue
                # to be scheduled again with the new start time (e.g. found start).
                # This process should reach its termination at least at the very end of this contractor's schedule.
                queue.extend(scheduled_wreqs)
                scheduled_wreqs.clear()
                scheduled_wreqs.append(wreq)
                start = max(start, found_start)

        # === THE INNER VALIDATION ===

        start_time = start
        for zone in zones:
            state = self._timeline[zone.kind]
            start_idx = state.bisect_right(start_time)
            end_idx = state.bisect_right(start_time + exec_time)
            start_status = state[start_idx - 1].available_workers_count

            # updating all events in between the start and the end of our current task
            for event in state[start_idx: end_idx]:
                # TODO Check that we shouldn't change the between statuses
                assert self._config.statuses.match_status(event.available_workers_count, zone.required_status)

            assert state[start_idx - 1].event_type == EventType.END \
                   or (state[start_idx - 1].event_type == EventType.START
                       and self._config.statuses.match_status(zone.required_status, start_status)) \
                   or state[start_idx - 1].event_type == EventType.INITIAL, \
                   f'{state[start_idx - 1].time} {state[start_idx - 1].event_type} {zone.required_status} {start_status}'

        # === END OF INNER VALIDATION ===

        return start

    def _match_status(self, target: int, match: int) -> bool:
        return self._config.statuses.match_status(target, match)

    def _find_earliest_time_slot(self,
                                 state: SortedList[ScheduleEvent],
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
            if i > 0 and i % 50 == 0:
                print(f'Warning! Probably cycle in looking for earliest time slot: {i} iteration')
                print(f'Current start time: {current_start_time}, current start idx: {current_start_idx}')
            i += 1
            end_idx = state.bisect_right(current_start_time + exec_time)

            # if we are inside the interval with wrong status
            # we should go right and search the best begin
            # if state[current_start_idx].event_type == EventType.START \
            #         and not self._match_status(state[current_start_idx].available_workers_count, required_status):
            #     current_start_idx += 1
            #     # if self._match_status(state[current_start_idx].available_workers_count, required_status)
            #     current_start_time = state[current_start_idx].time
            #     continue

            # here we are outside the all intervals or inside the interval with right status
            # if we are outside intervals, we can be in right or wrong status, so let's check it
            # else we are inside the interval with right status so let

            # we should count starts and ends on timeline prefix before the start_time
            # if starts_count is equal to ends_count, start_time is out of all the zone usage intervals
            # so we can change its status
            # starts_count = len([v for v in state[:current_start_idx + 1] if v.event_type == EventType.START])
            # ends_count = len([v for v in state[:current_start_idx + 1] if v.event_type == EventType.END])
            # if starts_count == ends_count \
            #         and not self._match_status(state[current_start_idx].available_workers_count, required_status):
            #     # we are outside all intervals, so let's decide should
            #     # we change zone status or go to the next checkpoint
            #     old_status = state[current_start_idx].available_workers_count
            #     # TODO Make this time calculation better: search the time slot for zone change before the start time
            #     change_cost = self._config.time_costs[old_status, required_status]
            #     prev_cpkt_idx = state.bisect_right(current_start_time - change_cost)
            #     if prev_cpkt_idx == current_start_idx or prev_cpkt_idx >= len(state):
            #         # we can change status before current_start_time
            #         start_time_changed = current_start_time
            #     else:
            #         start_time_changed = state[prev_cpkt_idx].time + 1 + change_cost # current_start_time + change_cost
            #
            #     next_cpkt_idx = min(current_start_idx + 1, len(state) - 1)
            #     next_cpkt_time = state[next_cpkt_idx].time
            #     if (parent_time <= next_cpkt_time <= start_time_changed
            #             and self._match_status(state[next_cpkt_idx].available_workers_count, required_status)):
            #         # waiting until the next checkpoint is faster that change zone status
            #         current_start_time = next_cpkt_time
            #         current_start_idx += 1
            #     else:
            #         current_start_time = start_time_changed
            #     # renewing the end index
            #     end_idx = state.bisect_right(current_start_time + exec_time)

            # here we are guaranteed that current_start_time is in right status
            # so go right and check matching statuses
            # this step performed like in MomentumTimeline
            not_compatible_status_found = False
            for idx in range(end_idx - 1, current_start_idx - 2, -1):
                if not self._match_status(state[idx].available_workers_count, required_status):
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
                cur_cpkt = state[-1]
                # if cur_cpkt.time == current_start_time and not self._match_status(cur_cpkt.available_workers_count,
                #                                                                   required_status):
                #     # print('Problem!')
                #     current_start_time = max(parent_time, state[-1].time + 1)
                current_start_time = max(parent_time, state[-1].time + 1)
                break

            current_start_time = state[current_start_idx].time

        # === THE INNER VALIDATION ===

        start_time = current_start_time
        start_idx = state.bisect_right(start_time)
        end_idx = state.bisect_right(start_time + exec_time)
        start_status = state[start_idx - 1].available_workers_count

        # updating all events in between the start and the end of our current task
        for event in state[start_idx: end_idx]:
            # TODO Check that we shouldn't change the between statuses
            assert self._config.statuses.match_status(event.available_workers_count, required_status)

        assert state[start_idx - 1].event_type == EventType.END \
               or (state[start_idx - 1].event_type == EventType.START
                   and self._config.statuses.match_status(start_status, required_status)) \
               or state[start_idx - 1].event_type == EventType.INITIAL, \
            f'{state[start_idx - 1].time} {state[start_idx - 1].event_type} {required_status} {start_status}'

        # === END OF INNER VALIDATION ===

        return current_start_time

    def update_timeline(self, index: int, zones: list[Zone], start_time: Time, exec_time: Time) -> list[ZoneTransition]:
        sworks = []

        for zone in zones:
            state = self._timeline[zone.name]
            start_idx = state.bisect_right(start_time)
            end_idx = state.bisect_right(start_time + exec_time)
            start_status = state[start_idx - 1].available_workers_count

            # updating all events in between the start and the end of our current task
            for event in state[start_idx: end_idx]:
                # TODO Check that we shouldn't change the between statuses
                assert self._config.statuses.match_status(event.available_workers_count, zone.status)

            assert state[start_idx - 1].event_type == EventType.END \
                   or (state[start_idx - 1].event_type == EventType.START
                       and self._config.statuses.match_status(zone.status, start_status)) \
                   or state[start_idx - 1].event_type == EventType.INITIAL, \
                   f'{state[start_idx - 1].time} {state[start_idx - 1].event_type} {zone.status} {start_status}'

            change_cost = self._config.time_costs[start_status, zone.status] \
                            if not self._config.statuses.match_status(zone.status, start_status) \
                            else 0

            state.add(ScheduleEvent(index, EventType.START, start_time - change_cost, None, zone.status))
            state.add(ScheduleEvent(index, EventType.END, start_time - change_cost + exec_time, None, zone.status))

            if start_status != zone.status and zone.status != 0:
                # if we need to change status, record it
                sworks.append(ZoneTransition(name=f'Access card {zone.name} status: {start_status} -> {zone.status}',
                                             from_status=start_status,
                                             to_status=zone.status,
                                             start_time=start_time - change_cost,
                                             end_time=start_time))
        return sworks
