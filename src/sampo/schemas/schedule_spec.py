from collections import defaultdict
from copy import copy
from typing import List, Dict, Optional, Union

from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.types import WorkerName
from sampo.schemas.works import WorkUnit


class WorkSpec:
    """
    Here are the container for externally given terms, that the resulting `ScheduledWork` should satisfy.
    Must be used in schedulers.
    :param chain: the chain of works, that should be scheduled one after another, e.g. inseparable,
    that starts from this work. Now unsupported.
    :param assigned_workers: predefined worker team (scheduler should assign this worker team to this work)
    :param assigned_time: predefined work time (scheduler should schedule this work with this execution time)
    """
    chain: Optional[List[WorkUnit]] = None  # TODO Add support
    assigned_workers: Dict[WorkerName, int] = {}
    assigned_time: Optional[Time] = None


class ScheduleSpec:
    """
    Here are the container for externally given terms, that Schedule should satisfy.
    Must be used in schedulers.
    :param work2spec: work specs
    """
    _work2spec: Dict[str, WorkSpec] = defaultdict(WorkSpec)

    def set_exec_time(self, work: Union[str, WorkUnit], time: Time) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_time = time
        return self

    def assign_workers_dict(self, work: str, workers: Dict[WorkerName, int]) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_workers = copy(workers)
        return self

    def assign_workers(self, work: str, workers: List[Worker]) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_workers = {worker.name: worker.count for worker in workers}
        return self

    def get_work_spec(self, work_id: str) -> WorkSpec:
        return self._work2spec[work_id]

