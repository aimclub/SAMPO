from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sampo.schemas.time import Time

ContractorName = str
WorkerName = str
AgentId = tuple[ContractorName, WorkerName]


# TODO check for the possibility of translation to IntEnum with the removal of priority
class EventType(Enum):
    INITIAL = -1
    START = 0
    END = 1

    @property
    def priority(self) -> int:
        """
        Returns the processing order for scheduling events

        :return value: The desired priority
        """
        # noinspection PyTypeChecker
        value = int(self.value)
        return value


@dataclass
class ScheduleEvent:
    seq_id: int
    event_type: EventType
    time: Time
    swork: Optional['ScheduledWork']
    available_workers_count: int
