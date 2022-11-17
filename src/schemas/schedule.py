from collections import defaultdict
from datetime import datetime
from functools import cached_property
from typing import Tuple, List, Dict, Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

from scheduler.utils.just_in_time_timeline import order_nodes_by_start_time
from schemas.graph import WorkGraph
from schemas.scheduled_work import ScheduledWork
from schemas.serializable import JSONSerializable, T
from schemas.time import Time
from schemas.works import WorkUnit
from utilities.datetime import timestamp_to_date, transform_timestamp
from utilities.schedule import fix_baps_tasks

ResourceSchedule = Dict[str, List[Tuple[Time, Time]]]
ScheduleWorkDict = Dict[str, ScheduledWork]


# TODO: describe the class (description, parameters)
class Schedule(JSONSerializable['Schedule']):
    _schedule: DataFrame
    _start: str

    _data_columns: List[str] = ['idx', 'task_id', 'task_name', 'contractor', 'volume',
                                'measurement', 'successors', 'start',
                                'finish', 'duration', 'workers']
    _scheduled_work_column: str = 'scheduled_work_object'

    _columns: List[str] = _data_columns + [_scheduled_work_column]

    # TODO: consider applying lru_cache to the properties
    # TODO: Rebase object onto ScheduleWorkDict and ordered ScheduledWork list
    # TODO: describe the function (return type)
    @property
    def full_schedule_df(self) -> DataFrame:
        """
        The full schedule DataFrame with all works, data columns and a distinct column for ScheduledWork objects
        """
        return self._schedule

    # TODO: describe the function (return type)
    @property
    def pure_schedule_df(self) -> DataFrame:
        """
        Schedule DataFrame without service units and containing only original columns (stored in _data_columns field)
        """
        return self._schedule[~self._schedule.apply(
            lambda row: row[self._scheduled_work_column].work_unit.is_service_unit,
            axis=1
        )][self._data_columns]

    # TODO: describe the function (description, return type)
    @cached_property
    def merged_stages_datetime_df(self) -> DataFrame:
        result = fix_baps_tasks(self._schedule)
        result.start = [self.time_in_schedule_to_date(t) for t in result.start]
        result.finish = [self.time_in_schedule_to_date(t) for t in result.finish]
        return result

    # TODO: describe the function (description, return type)
    @property
    def works(self) -> Iterable[ScheduledWork]:
        return self._schedule.scheduled_work_object

    # TODO: describe the function (description, return type)
    @property
    def to_schedule_work_dict(self) -> ScheduleWorkDict:
        return {r['task_id']: r['scheduled_work_object'] for _, r in self._schedule.iterrows()}

    # TODO: describe the function (description, return type)
    @property
    def execution_time(self) -> Time:
        return self._schedule.iloc[-1].finish

    # TODO: describe the function (description, parameters, return type)
    def __init__(self, schedule: DataFrame, start: str) -> None:
        self._schedule = schedule
        self._start = start

    # TODO: describe the function (description, return type)    
    def _serialize(self) -> T:
        return {
            'works': [sw._serialize() for sw in self._schedule.scheduled_work_object],
            'start': self._start
        }

    # TODO: describe the function (description, parameters, return type)
    @classmethod
    def _deserialize(cls, dict_representation: T) -> 'Schedule':
        dict_representation['works'] = [ScheduledWork._deserialize(sw) for sw in dict_representation['works']]
        return Schedule.from_scheduled_works(**dict_representation)

    # TODO: describe the function (description, parameters, return type)
    def time_in_schedule_to_date(self, time: Time) -> datetime:
        return timestamp_to_date(transform_timestamp(time, self._start))

    @staticmethod
    def from_scheduled_works(works: Iterable[ScheduledWork],
                             start: str,
                             wg: WorkGraph = None) \
            -> 'Schedule':
        """
        Factory method to create a Schedule object from list of Schedule works and additional info
        :param wg:
        :param works:
        :param work_info: Info about works or path to file with it
        :param start: Start of schedule
        :return: Schedule
        """
        ordered_task_ids = order_nodes_by_start_time(works, wg) if wg else None

        def info(work_unit: WorkUnit) -> Tuple[float, str, List[Tuple[str, str]]]:
            if not wg:
                return 0, "", []
            # noinspection PyTypeChecker
            return work_unit.volume, work_unit.volume_type, \
                   [(edge.finish.id, edge.type.value) for edge in wg[work_unit.id].edges_from]

        # TODO: describe the function (description, parameters, return type)
        # start, end, duration
        def sed(t1, t2) -> tuple:
            # s, e = tuple(sorted([timestamp_to_date(transform_timestamp(t, start)) for t in (t1, t2)]))
            s, e = tuple(sorted((t1, t2)))
            return s, e, e - s

        df = [(i,  # idx
               w.work_unit.id,  # task_id
               w.work_unit.name,  # task_name
               w.contractor,  # contractor info
               *info(w.work_unit),  # volume, measurement, successors
               *sed(*(t.value for t in w.start_end_time)),  # start, end, duration
               repr(dict((i.name, i.count) for i in w.workers)),  # workers
               w  # full ScheduledWork info
               ) for i, w in enumerate(works)]
        df = DataFrame.from_records(df, columns=Schedule._columns)

        df = df.set_index('idx')

        if ordered_task_ids:
            df.task_id = df.task_id.astype('category')
            df.task_id = df.task_id.cat.set_categories(ordered_task_ids)
            df = df.sort_values(['task_id'])
            df.task_id = df.task_id.astype(str)

        df = df.reindex(columns=Schedule._columns)
        df = df.reset_index(drop=True)

        return Schedule(df, start)
