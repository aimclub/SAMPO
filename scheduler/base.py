from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from schemas.contractor import Contractor
from schemas.schedule import Schedule
from schemas.graph import WorkGraph

TIME_SHIFT = 0.05


class SchedulerType(Enum):
    Topological = 'topological'
    Evolutionary = 'evolution'


class Scheduler(ABC):
    scheduler_type: SchedulerType

    @abstractmethod
    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 ksg_info: Dict[str, Any],
                 start: str,
                 validate_schedule: Optional[bool] = True) \
            -> Tuple[Schedule, List[str]]:
        ...
