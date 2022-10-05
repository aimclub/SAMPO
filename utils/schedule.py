import pandas as pd
from typing import Dict, Any

from schemas.time import Time


def prepare_ksg_info_dictionary(ksg_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Create dictionary with information about volume, measurement unit and successors ids
    for all tasks of given brksg
    :param ksg_file_path:
    :return: dict with tasks ids as keys and info dicts as values
    """
    ksg_df = pd.read_csv(ksg_file_path, sep=';',
                         dtype={'activity_id': str, 'predecessor_ids': str, 'connection_types': str})
    # Initialize dictionary, add info about task's volume and measurement unit
    id2info = {}

    for i in ksg_df.index:
        task_id = ksg_df.loc[i, 'activity_id']
        id2info[task_id] = {}
        id2info[task_id]['successors'] = []
        id2info[task_id]['volume'] = ksg_df.loc[i, 'volume']
        id2info[task_id]['measurement'] = ksg_df.loc[i, 'measurement']

    # Add info about task's successors
    for i in ksg_df.index:
        # get list of all predecessors' ids for the current task
        pred_ids = str(ksg_df.loc[i, 'predecessor_ids']).split(',')
        conn_types = str(ksg_df.loc[i, 'connection_types']).split(',')
        for pred_id, conn_type in zip(pred_ids, conn_types):
            if pred_id != 'nan':
                if pred_id in id2info:
                    id2info[pred_id]['successors'].append([ksg_df.loc[i, 'activity_id'],
                                                           conn_type])
    return id2info


def fix_baps_tasks(baps_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge information for all tasks, which were separated on the several stages during baps
    :param baps_schedule_df: pd.DataFrame: schedule with info for tasks separated on stages
    :return: pd.DataFrame: schedule with merged info for all real tasks
    """
    df_len = baps_schedule_df.shape[0]
    baps_schedule_df.index = range(df_len)

    df = pd.DataFrame(columns=baps_schedule_df.columns)
    unique_ids = set([x.split('_')[0] for x in baps_schedule_df.task_id])

    for task_id in unique_ids:
        task_stages_df = baps_schedule_df.loc[baps_schedule_df.loc[:, 'task_id'].str.contains(task_id)]
        task_series = merge_baps_stages(task_stages_df.reset_index(drop=True))
        df.loc[df.shape[0]] = task_series  # append

    df = df.sort_values(by=['start', 'task_name'])
    df['idx'] = range(len(df))

    return df


def merge_baps_stages(task_df: pd.DataFrame) -> pd.Series:
    """
    Merge baps stages of the same real task into one
    :param task_df: pd.DataFrame: one real task's stages dataframe, sorted by start time
    :return: pd.Series with the full information about the task
    """
    if len(task_df) == 1:
        return task_df.loc[0, :]
    else:
        df = task_df.iloc[-1:].reset_index(drop=True)
        for column in ['task_id', 'task_name']:
            df.loc[0, column] = df.loc[0, column].split('_')[0]  # fix task id and name
        # fix task's start time and duration
        df.loc[0, 'start'] = task_df.loc[0, 'start']
        df.loc[0, 'duration'] = df.loc[0, 'finish'] - df.loc[0, 'start'] + Time(1)
        return df.loc[0, :]
