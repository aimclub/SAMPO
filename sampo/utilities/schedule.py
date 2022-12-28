import pandas as pd


def fix_split_tasks(baps_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge information for all tasks, which were separated on the several stages during split

    :param baps_schedule_df: pd.DataFrame: schedule with info for tasks separated on stages
    :return: pd.DataFrame: schedule with merged info for all real tasks
    """
    df_len = baps_schedule_df.shape[0]
    baps_schedule_df.index = range(df_len)

    df = pd.DataFrame(columns=baps_schedule_df.columns)
    unique_ids = set([x.split('_')[0] for x in baps_schedule_df.task_id])

    for task_id in unique_ids:
        task_stages_df = baps_schedule_df.loc[baps_schedule_df.loc[:, 'task_id'].str.contains(task_id)]
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
    if len(task_df) == 1:
        df = task_df.copy()
        df['successors'] = [[tuple([x[0].split('_')[0], x[1]]) for x in df.loc[0, 'successors']]]
        return df.loc[0, :]
    else:
        df = task_df.copy()
        df = df.iloc[-1:].reset_index(drop=True)
        for column in ['task_id', 'task_name']:
            df.loc[0, column] = df.loc[0, column].split('_')[0]  # fix task id and name

        # sum up volumes through all stages
        df.loc[0, 'volume'] = sum(task_df.loc[:, 'volume'])
        df.loc[0, 'workers'] = task_df.loc[0, 'workers']

        # fix connections through all stages
        fixed_connections_lst = []
        for connections_lst in task_df.loc[:, 'successors']:
            for connection in connections_lst:
                if connection[1] != 'IFS':
                    fixed_connections_lst.append(tuple([connection[0].split('_')[0], connection[1]]))
        fixed_connections_lst = list(set(fixed_connections_lst))
        df.loc[:, 'successors'] = [fixed_connections_lst]

        # fix task's start time and duration
        df.loc[0, 'start'] = task_df.loc[0, 'start']
        df.loc[0, 'finish'] = task_df.loc[len(task_df) - 1, 'finish']
        df.loc[0, 'duration'] = (df.loc[0, 'finish'] - df.loc[0, 'start']).days + 1

        return df.loc[0, :]


def remove_service_tasks(service_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'start', 'finish' and milestone tasks from the schedule

    :param service_schedule_df: pd.DataFrame: schedule (with merges stages in the case of baps) with service tasks
    :return: pd.DataFrame: schedule without information about service tasks
    """
    schedule_df = service_schedule_df.copy()

    service_df = schedule_df.loc[:, 'task_name'].str.contains('start|finish')

    # Prepare list with service tasks ids
    service_tasks_ids = set(schedule_df.loc[service_df].loc[:, 'task_id'])

    # Remove rows with service tasks from DataFrame
    schedule_df = schedule_df.loc[~service_df]

    # Fix connections linked to the service tasks
    fixed_connections_lst = []
    for connections_lst in schedule_df.loc[:, 'successors']:
        fixed_connections_lst.append([])
        for connection in connections_lst:
            if connection[0] not in service_tasks_ids:
                fixed_connections_lst[-1].append(connection)
    schedule_df.loc[:, 'successors'] = fixed_connections_lst
    return schedule_df