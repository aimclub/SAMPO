from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.types import WorkerName
from sampo.schemas.works import WorkUnit


@dataclass
class WorkSpec:
    """
    Here are the container for externally given terms, that the resulting `ScheduledWork` should satisfy.
    Must be used in schedulers.
    :param chain: the chain of works, that should be scheduled one after another, e.g. inseparable,
    that starts from this work. Now unsupported.
    :param assigned_workers: predefined worker team (scheduler should assign this worker team to this work)
    :param assigned_time: predefined work time (scheduler should schedule this work with this execution time)
    :param is_independent: should this work be resource-independent, e.g. executing with no parallel users of
    its types of resources
    """
    chain: list[WorkUnit] | None = None  # TODO Add support
    assigned_workers: dict[WorkerName, int] = field(default_factory=dict)
    assigned_time: Time | None = None
    is_independent: bool = False

@dataclass
class ScheduleSpec:
    """
    Here is the container for externally given terms, that Schedule should satisfy.
    Must be used in schedulers.

    :param work2spec: work specs
    """

    _work2spec: dict[str, WorkSpec] = field(default_factory=lambda: defaultdict(WorkSpec))

    def set_exec_time(self, work: str | WorkUnit, time: Time) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_time = time
        return self

    def assign_workers_dict(self, work: str, workers: dict[WorkerName, int]) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_workers = copy(workers)
        return self

    def assign_workers(self, work: str, workers: list[Worker]) -> 'ScheduleSpec':
        if isinstance(work, WorkUnit):
            work = work.id
        self._work2spec[work].assigned_workers = {worker.name: worker.count for worker in workers}
        return self

    def get_work_spec(self, work_id: str) -> WorkSpec:
        return self._work2spec[work_id]
