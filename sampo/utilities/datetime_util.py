from datetime import datetime, timedelta
from typing import Union, Optional

from sampo.schemas.time import Time


def ftime(dt: datetime, date_format: str = '%y-%m-%d %H:%M:%S') -> str:
    return dt.strftime(date_format)


def parse_datetime(dts: str, date_format: Optional[str] = None) -> datetime:
    """
    Parses datetime from string

    :param dts: String datetime
    :param date_format: String format. If not provided, '%Y-%m-%d' and then '%y-%m-%d %H:%M:%S' are tried.
    :return:
    """
    if date_format is None:
        try:
            return datetime.strptime(dts, '%Y-%m-%d')
        except ValueError:
            return datetime.strptime(dts, '%y-%m-%d %H:%M:%S')
    return datetime.strptime(dts, date_format)


def add_time_delta(base_datetime: Union[datetime, str],
                   time_delta: Union[Time, float],
                   time_units: Optional[str] = 'days') -> datetime:
    """
    Adds time delta to base datetime

    :param base_datetime:
    :param time_delta:
    :param time_units: can be days, seconds, microseconds, milliseconds, minutes, hours, weeks
    :return:
    """
    base = parse_datetime(base_datetime) if isinstance(base_datetime, str) else base_datetime
    delta = timedelta(**{time_units: int(time_delta.value if isinstance(time_delta, Time) else time_delta)})

    return base + delta
