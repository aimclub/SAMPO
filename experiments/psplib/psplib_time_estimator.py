from sampo.schemas.resources import Worker
from sampo.schemas.resources import WorkerProductivityMode
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit
from sampo.schemas.time_estimator import WorkTimeEstimator, WorkEstimationMode


class PSPlibWorkTimeEstimator(WorkTimeEstimator):
    def __init__(self, times: dict[str, Time]):
        self.times = dict(times)

    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> dict[str, int]:
        return dict()

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        return

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        return

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        return Time(self.times.get(work_unit.id, 0))

# to avoid changing PSPlibWorkTimeEstimator for now
class ConstantWorkTimeEstimator(WorkTimeEstimator):
    """Helper class for a constant activity time"""
    
    def __init__(self, job_durations: dict[str, int]):
        self.times = {
            # ensure correct data types
            str(job_id): Time(time_int)
            for job_id, time_int in job_durations.items()
        }

    def estimate_time(self, work_unit: WorkUnit, *args):
        estimate = self.times[work_unit.id]
        return Time(estimate)

    def find_work_resources(self, *args):
        return dict()

    def set_estimation_mode(self, *args):
        return None

    def set_productivity_mode(self, *args):
        return None
    
    def get_recreate_info(self, *args):
        return None