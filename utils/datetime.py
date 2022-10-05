import time
from datetime import datetime, timedelta
from typing import Tuple, Union

import pandas as pd

from schemas.time import Time


def ftime(dt: datetime, f: str = '%y-%m-%d %H:%M:%S') -> str:
    return dt.strftime(f)


def transform_timestamp(schedule_time: Union[float, Time], start: str, time_unit='day') -> float:
    """
    Transform timestamp according to the given time unit and start date
    :param schedule_time: original timestamp
    :param start: str start date in the format "%Y-%m-%d"
    :param time_unit: unit of time for the timestamp: 'day' or 'hour' [otherwise]
    :return: updated timestamp
    """
    schedule_time = schedule_time.value if isinstance(schedule_time, Time) else schedule_time
    start_timestamp = time.mktime(datetime.strptime(start, "%Y-%m-%d").timetuple())
    return (schedule_time * 60 * 60 * 24 if time_unit == 'day' else schedule_time * 60 * 60) + start_timestamp


def timestamp_to_date(timestamp: float) -> datetime:
    """
    Convert timestamp to the date and time
    :param timestamp: original timestamp
    :return: date and time from timestamp
    """
    return datetime.fromtimestamp(timestamp)


def timedelta_to_days(td: timedelta) -> int:
    """
    Round given timedelta to days
    :param td: original timestamp
    :return: date and time from timestamp
    """
    return td.days


def fix_time_order(times_df: pd.DataFrame, right_column_order: Tuple[str, str] = ('start', 'finish')) -> pd.DataFrame:
    """
    Fix order of the start and finish for the milestone timestamps in the schedule
    :param times_df: DataFrame with two columns 'start' and 'finish' timestamps of the scheduled tasks
    :param right_column_order: Names of columns, which contain times,
    that should be ordered with respect to the column order
    :return: DataFrame with fixed timestamp columns
    """
    assert len(right_column_order) == 2, 'Wrong column order. Should be 2 existing columns.'
    start, finish = right_column_order

    df = pd.DataFrame(columns=[start, finish])
    df.loc[:, start] = times_df.min(axis=1)
    df.loc[:, finish] = times_df.max(axis=1)

    # print(df)

    time_delta = df.loc[0, finish] - df.loc[0, start]
    for i in range(1, len(df.loc[:, start])):
        df.loc[i, start] = df.loc[i, start] - i * time_delta
        df.loc[i, finish] = df.loc[i, 'finish'] - i * time_delta
    return df

# def remove_time_delta(times_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Remove technical time delta (necessary for the algorithms) for all tasks
#     :param times_df: DataFrame with two columns 'start' and 'finish' timestamps of the scheduled tasks
#     :return: DataFrame with fixed timestamp columns
#     """
#     times_df = times_df.reset_index()
