from typing import TypeVar, Generic

from sortedcontainers import SortedList

from sampo.schemas.time import Time
from sampo.schemas.types import EventType

T = TypeVar('T')


class GeneralTimeline(Generic[T]):
    """
    The representation of general-purpose timeline that supports some general subset of functions
    """
    def __init__(self):
        # ScheduleEvent = time, idx, object
        def event_cmp(event: Time | tuple[EventType, Time, int, T]) -> tuple[Time, int, int]:
            if isinstance(event, tuple):
                return event[1], event[2], event[0].priority

            if isinstance(event, Time):
                # instances of Time must be greater than almost all ScheduleEvents with same time point
                return event, Time.inf().value, 2

            raise ValueError(f'Incorrect type of value: {type(event)}')

        self._timeline = SortedList(iterable=((EventType.INITIAL, Time(0), -1, None),), key=event_cmp)
        self._next_idx = 0

    def update_timeline(self, start_time: Time, exec_time: Time, obj: T):
        self._timeline.add((EventType.START, start_time, self._next_idx, obj))
        self._timeline.add((EventType.END, start_time + exec_time, self._next_idx, obj))
        self._next_idx += 1

    def __getitem__(self, index) -> Time:
        """
        Returns the time of checkpoint on `index`
        """
        return self._timeline[index][1]

    def __len__(self):
        return len(self._timeline)
