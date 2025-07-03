from typing import Type

from sampo.schemas.resources import Worker
from sampo.schemas.resources import WorkerProductivityMode
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit
from sampo.schemas.time_estimator import WorkTimeEstimator, WorkEstimationMode


class PSPlibWorkTimeEstimator(WorkTimeEstimator):
    def __init__(self, times: dict[str, Time]):
        self.times = dict(times)

    def find_work_resources(self, work_name: str, work_volume: float,
                            resource_name: list[str] | None = None,
                            measurement: str | None = None) -> dict[str, int]:
        return dict()

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        return

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        return

    def get_recreate_info(self) -> tuple[Type, tuple]:
        return PSPlibWorkTimeEstimator, ()

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        return Time(self.times.get(work_unit.id, 0))

    def get_recreate_info(self):
        return PSPlibWorkTimeEstimator, ()

