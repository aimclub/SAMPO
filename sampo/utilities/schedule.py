from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from pandas import DataFrame

from sampo.structurator import STAGE_SEP
from sampo.utilities.datetime_util import add_time_delta


def offset_schedule(schedule: DataFrame, offset: datetime | str) -> DataFrame:
    """
    Returns full schedule object with `start` and `finish` columns pushed by date in `offset` argument.
    :param schedule: the schedule itself
    :param offset: Start of schedule, to add as an offset.
    :return: Shifted schedule DataFrame.
    """
    r = schedule.loc[:, :]
    r['start_offset'] = r['start'].apply(partial(add_time_delta, offset))
    r['finish_offset'] = r['finish'].apply(partial(add_time_delta, offset))
    r = r.rename({'start': 'start_', 'finish': 'finish_',
                  'start_offset': 'start', 'finish_offset': 'finish'}, axis=1) \
        .drop(['start_', 'finish_'], axis=1)
    return r


def fix_split_tasks(baps_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge information for all tasks, which were separated on the several stages during split

    :param baps_schedule_df: pd.DataFrame: schedule with info for tasks separated on stages
    :return: pd.DataFrame: schedule with merged info for all real tasks
    """
    df_len = baps_schedule_df.shape[0]
    baps_schedule_df.index = range(df_len)

    df = pd.DataFrame(columns=baps_schedule_df.columns)
    unique_ids = {x.split(STAGE_SEP)[0] for x in baps_schedule_df.task_id}

    for task_id in unique_ids:
        task_stages_df = baps_schedule_df.loc[
            baps_schedule_df.task_id.str.startswith(f'{task_id}{STAGE_SEP}')
            | (baps_schedule_df.task_id == task_id)
            ]
        task_series = merge_split_stages(task_stages_df.reset_index(drop=True))
        df.loc[df.shape[0]] = task_series  # append

    df = df.sort_values(by=['start', 'task_name'])
    df['idx'] = range(len(df))

    return df


def merge_split_stages(task_df: pd.DataFrame) -> pd.Series:
    """
    Merge split stages of the same real task into one

    :param task_df: pd.DataFrame: one real task's stages dataframe, sorted by start time
    :return: pd.Series with the full information about the task
    """

    def get_stage_num(name: str):
        split_name = name.split(STAGE_SEP)
        return int(split_name[-1]) if len(split_name) > 1 else -1

    if len(task_df) > 1:
        df = task_df.copy()
        df['stage_num'] = df['task_name_mapped'].apply(get_stage_num)
        df = df.sort_values(by='stage_num')
        df = df.reset_index(drop=True)

        df = df.iloc[-1:].reset_index(drop=True)
        for column in ['task_id', 'task_name', 'task_name_mapped']:
            df.loc[0, column] = df.loc[0, column].split(STAGE_SEP)[0]  # fix task id and name

        # sum up volumes through all stages
        df.loc[0, 'volume'] = sum(task_df.loc[:, 'volume'])
        df.loc[0, 'workers'] = task_df.loc[0, 'workers']

        # fix task's start time and duration
        df.loc[0, 'start'] = task_df.loc[0, 'start']
        df.loc[0, 'finish'] = task_df.loc[len(task_df) - 1, 'finish']
        if isinstance(df.loc[0, 'start'], np.int64) or isinstance(df.loc[0, 'start'], np.int32):
            df.loc[0, 'duration'] = df.loc[0, 'finish'] - df.loc[0, 'start'] + 1
        else:
            df.loc[0, 'duration'] = (df.loc[0, 'finish'] - df.loc[0, 'start']).days + 1
    else:
        df = task_df.copy()

    return df.loc[0, :]
