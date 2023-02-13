from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

from sampo.schemas.time import Time

ContractorName = str
WorkerName = str
AgentId = Tuple[ContractorName, WorkerName]


# TODO check for the possibility of translation to IntEnum with the removal of priority
# TODO: describe the class (description)
class EventType(Enum):
    Initial = -1
    Start = 0
    End = 1

    @property
    def priority(self) -> int:
        """
        Returns the processing order for scheduling events

        :return value: The desired priority
        """
        # noinspection PyTypeChecker
        value = int(self.value)
        return value


# TODO: describe the class (description, parameters)
@dataclass
class ScheduleEvent:
    seq_id: int
    event_type: EventType
    time: Time
    swork: Optional['ScheduledWork']
    available_workers_count: int
