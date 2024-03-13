import math
from itertools import chain
from operator import attrgetter
from random import Random
from sampo.schemas.time import Time
from typing import Type

from business_calendar import Calendar
from datetime import datetime, timedelta, time

from sampo.utilities.collections_util import build_index
from sampo.schemas import WorkTimeEstimator, WorkUnit, Worker, WorkerReq, WorkEstimationMode, WorkerProductivityMode


def get_date_by_start_time(project_start_date, working_hours, start_time):
        days_from_start = int(start_time.value // working_hours)
        task_start_date = project_start_date + timedelta(days_from_start)
        return task_start_date.date()


def get_calendar_hours_from_working_hours(calendar, start_date, work_execution_hours, working_hours):
    working_days = int(math.ceil(work_execution_hours / working_hours))
    additional_working_hours = work_execution_hours % working_hours
    task_start_date = datetime.combine(start_date, time())
    calendar_end_date = calendar.addbusdays(task_start_date - timedelta(days=1), working_days)
    calendar_duration = calendar_end_date - task_start_date + timedelta(days=1)
    return calendar_duration.days * working_hours + additional_working_hours


class WorkEstimator(WorkTimeEstimator):

    def __init__(self):
        self._use_idle = True
        self._estimation_mode = WorkEstimationMode.Realistic
        self._productivity_mode = WorkerProductivityMode.Static

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker], start_time: Time | None = None):
        work_volume = work_unit.volume

        if work_unit.is_service_unit:
            return Time(0)
        else:
            cnt_workers = 0
            for worker in worker_list:
                cnt_workers += worker.count
            if cnt_workers != 0:
                work_execution_time = int(math.ceil(work_volume / cnt_workers))
                return Time(work_execution_time)
            else:
                return Time.inf()

    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> list[WorkerReq]:
        return []

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._estimation_mode = mode

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        self._productivity_mode = mode

    def get_recreate_info(self) -> tuple[Type, tuple]:
        return WorkEstimator, ()


class CalendarBasedWorkEstimator(WorkTimeEstimator):

    def __init__(self,
                 project_calendar: Calendar,
                 working_hours_cnt: int,
                 start_date: datetime):
        self._use_idle = True
        self._estimation_mode = WorkEstimationMode.Realistic
        self.calendar = project_calendar
        self.working_hours = working_hours_cnt
        self.project_start_date = start_date
        self._productivity_mode = WorkerProductivityMode.Static

    def get_date_by_start_time(self, start_time: Time):
        days_from_start = int(start_time.value // self.working_hours)
        task_start_date = self.project_start_date + timedelta(days_from_start)
        return task_start_date.date()

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker], start_time: Time | None = None):
        work_volume = work_unit.volume

        if work_unit.is_service_unit:
            return Time(0)
        else:
            cnt_workers = 0
            for worker in worker_list:
                cnt_workers += worker.count
            work_execution_time = int(math.ceil(work_volume / cnt_workers)) # in hours
            
            if start_time is None:
                return Time(work_execution_time)
                
            task_start_date = get_date_by_start_time(self.project_start_time, 
                                                     self.working_hours, 
                                                     start_time) # get task start date from Time attribute value
            calendar_duration = get_calendar_hours_from_working_hours(self.calendar, 
                                                                      task_start_date, 
                                                                      work_execution_time, 
                                                                      self.working_hours)
            return Time(calendar_duration)
            

    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> list[WorkerReq]:
        return []

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._estimation_mode = mode

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        self._productivity_mode = mode

    def get_recreate_info(self) -> tuple[Type, tuple]:
        return WorkEstimator, (self.calendar, self.working_hours, self.project_start_date)
