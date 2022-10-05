from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Tuple, List, Optional, Dict, Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

from schemas.serializable import AutoJSONSerializable, JSONSerializable, T
from schemas.time import Time
from schemas.resources import Worker, Equipment, ConstructionObject
from schemas.works import WorkUnit
from utils.datetime import timestamp_to_date, transform_timestamp
from utils.schedule import fix_baps_tasks
from utils.serializers import custom_serializer


@dataclass
class ScheduledWork(AutoJSONSerializable['ScheduledWork']):

    work_unit: WorkUnit
    start_end_time: Tuple[Time, Time]
    workers: List[Worker]
    equipments: Optional[List[Equipment]] = None
    materials: Optional[List[Equipment]] = None
    object: Optional[ConstructionObject] = None

    ignored_fields = ['equipments', 'materials', 'object']

    @custom_serializer('workers')
    @custom_serializer('start_end_time')
    def serialize_serializable_list(self, value):
        return [t._serialize() for t in value]

    @classmethod
    @custom_serializer('start_end_time', deserializer=True)
    def deserialize_time(cls, value):
        return [Time._deserialize(t) for t in value]

    @classmethod
    @custom_serializer('workers', deserializer=True)
    def deserialize_workers(cls, value):
        return [Worker._deserialize(t) for t in value]

    @property
    def start_time(self) -> Time:
        return self.start_end_time[0]

    @property
    def finish_time(self) -> Time:
        return self.start_end_time[1]

    @staticmethod
    def start_time_getter():
        return lambda x: x.start_end_time[0]

    @staticmethod
    def finish_time_getter():
        return lambda x: x.start_end_time[1]

    @property
    def duration(self) -> Time:
        start, end = self.start_end_time
        return end - start

    def is_overlapped(self, time: int) -> bool:
        start, end = self.start_end_time
        return start <= time < end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.work_unit.id,
            "task_name": self.work_unit.name,
            "start": self.start_time.value,
            "finish": self.finish_time.value,
            "workers": {worker.name: worker.count for worker in self.workers},
        }


ResourceSchedule = Dict[str, List[Tuple[Time, Time]]]
ScheduleWorkDict = Dict[str, ScheduledWork]
ScheduleOrderDict = Dict[str, Tuple[ScheduleWorkDict, List[str]]]


class Schedule(JSONSerializable['Schedule']):
    _schedule: DataFrame
    _start: str

    _data_columns: List[str] = ['idx', 'task_id', 'task_name', 'volume',
                                'measurement', 'successors', 'start',
                                'finish', 'duration', 'workers']
    _scheduled_work_column: str = 'scheduled_work_object'

    _columns: List[str] = _data_columns + [_scheduled_work_column]

    # TODO: consider applying lru_cache to the properties
    # TODO: Rebase object onto ScheduleWorkDict and ordered ScheduledWork list
    @property
    def full_schedule_df(self) -> DataFrame:
        """
        The full schedule DataFrame with all works, data columns and a distinct column for ScheduledWork objects
        """
        return self._schedule

    @property
    def pure_schedule_df(self) -> DataFrame:
        """
        Schedule DataFrame without service units and containing only original columns (stored in _data_columns field)
        """
        return self._schedule[~self._schedule.apply(
            lambda row: row[self._scheduled_work_column].work_unit.is_service_unit,
            axis=1
        )][self._data_columns]

    @cached_property
    def merged_stages_datetime_df(self) -> DataFrame:
        result = fix_baps_tasks(self._schedule)
        result.start = [self.time_in_schedule_to_date(t) for t in result.start]
        result.finish = [self.time_in_schedule_to_date(t) for t in result.finish]
        return result

    @property
    def works(self) -> Iterable[ScheduledWork]:
        return self._schedule.scheduled_work_object

    @property
    def to_schedule_work_dict(self) -> ScheduleWorkDict:
        return {r['task_id']: r['scheduled_work_object'] for _, r in self._schedule.iterrows()}

    @property
    def execution_time(self) -> Time:
        return self._schedule.iloc[-1].finish

    def __init__(self, schedule: DataFrame, start: str):
        self._schedule = schedule
        self._start = start

    def _serialize(self) -> T:
        return {
            'works': [sw._serialize() for sw in self._schedule.scheduled_work_object],
            'work_info': {r[1].task_id: {
                'measurement': r[1].measurement,
                'volume': r[1].volume,
                'successors': r[1].successors
            } for r in self._schedule.iterrows()},
            'start': self._start
        }

    @classmethod
    def _deserialize(cls, dict_representation: T) -> 'Schedule':
        dict_representation['works'] = [ScheduledWork._deserialize(sw) for sw in dict_representation['works']]
        return Schedule.from_scheduled_works(**dict_representation)

    def time_in_schedule_to_date(self, time: Time) -> datetime:
        return timestamp_to_date(transform_timestamp(time, self._start))

    @staticmethod
    def from_scheduled_works(works: Iterable[ScheduledWork],
                             work_info: Dict[str, Dict[str, Any]],
                             start: str) \
            -> 'Schedule':
        """
        Factory method to create a Schedule object from list of Schedule works and additional info
        :param works:
        :param work_info: Info about works or path to file with it
        :param start: Start of schedule
        :return: Schedule
        """
        work_info = work_info if isinstance(work_info, defaultdict) else defaultdict(lambda: None, work_info)

        def info(w: ScheduledWork) -> tuple:
            i = work_info[w.work_unit.id]
            if i:
                m = i['measurement']
                m = "" if m in [None, float('nan'), np.nan] or m is pd.NA else m
                return i['volume'], m, i['successors']
            return 0, "", []

        # start, end, duration
        def sed(t1, t2) -> tuple:
            # s, e = tuple(sorted([timestamp_to_date(transform_timestamp(t, start)) for t in (t1, t2)]))
            s, e = tuple(sorted((t1, t2)))
            return s, e, e - s

        df = [(i,  # idx
               w.work_unit.id,  # task_id
               w.work_unit.name,  # task_name
               *info(w),  # volume, measurement, successors
               *sed(*(t.value for t in w.start_end_time)),  # start, end, duration
               repr(dict((i.name, i.count) for i in w.workers)),  # workers
               w  # full ScheduledWork info
               ) for i, w in enumerate(works)]
        df = DataFrame.from_records(df, columns=Schedule._columns)

        df = df.set_index('idx')
        df = df.sort_values(by=['start', 'task_name'])  # , ascending=False)
        df = df.reindex(columns=Schedule._columns)
        df = df.reset_index(drop=True)

        return Schedule(df, start)
