import pandas as pd

from sampo.structurator import STAGE_SEP


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
    if len(task_df) > 1:
        task_df = task_df.sort_values(by='task_name_mapped')
        task_df = task_df.reset_index(drop=True)
        df = task_df.copy()

        df = df.iloc[-1:].reset_index(drop=True)
        for column in ['task_id', 'task_name']:
            df.loc[0, column] = df.loc[0, column].split(STAGE_SEP)[0]  # fix task id and name

        # sum up volumes through all stages
        df.loc[0, 'volume'] = sum(task_df.loc[:, 'volume'])
        df.loc[0, 'workers'] = task_df.loc[0, 'workers']

        # fix task's start time and duration
        df.loc[0, 'start'] = task_df.loc[0, 'start']
        df.loc[0, 'finish'] = task_df.loc[len(task_df) - 1, 'finish']
        df.loc[0, 'duration'] = (df.loc[0, 'finish'] - df.loc[0, 'start']).days + 1
    else:
        df = task_df.copy()

    return df.loc[0, :]
