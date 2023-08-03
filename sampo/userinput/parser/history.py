from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from sampo.utilities.task_name import NameMapper


def get_all_connections(tasks_df: pd.DataFrame, use_mapper: bool = False, mapper: NameMapper = None) \
        -> Tuple[dict[str, list], dict[str, list]]:
    """
    Return all connections from DataFrame of works
    """

    task_name_column = 'activity_name'
    if 'granular_name' in tasks_df:
        task_name_column = 'granular_name'

    num_tasks = len(tasks_df)
    # Get the upper triangular indices to avoid duplicate pairs
    indices = np.triu_indices(num_tasks, k=1)
    works1_ids = tasks_df['activity_id'].values[indices[0]]
    works1_names = tasks_df[task_name_column].values[indices[0]]

    works2_ids = tasks_df['activity_id'].values[indices[1]]
    works2_names = tasks_df[task_name_column].values[indices[1]]

    if use_mapper:
        works1_names = np.vectorize(mapper.get)(works1_names)
        works2_names = np.vectorize(mapper.get)(works2_names)

    return {"ids": works1_ids, "names": works1_names}, {"ids": works2_ids, "names": works2_names}


def get_delta_between_dates(first: str, second: str) -> int:
    return max((datetime.date(int(first.split('-')[0]), int(first.split('-')[1]), int(first.split('-')[2])) -
                datetime.date(int(second.split('-')[0]), int(second.split('-')[1]), int(second.split('-')[2]))).days, 1)


def find_min_without_outliers(lst: list[float]) -> float:
    return round(min([x for x in lst if x >= np.mean(lst) - 3 * np.std(lst)]), 2)


def gather_links_types_statistics(s1: str, f1: str, s2: str, f2: str) \
        -> Tuple[int, int, int, list, list, int, list, list, int, list, list, int, list, list]:
    """
    Count statistics on the occurrence of different mutual arrangement of tasks

    :param s1: start of first work
    :param f1: finish of first work
    :param s2: start of second work
    :param f2: finish of second work
    :return: Statistics on the occurrence of different mutual arrangement of tasks
    """

    fs12, fs21, ss12, ss21 = 0, 0, 0, 0
    ss12_lags, ss12_percent_lags, ss21_lags, ss21_percent_lags = [], [], [], []

    ffs12, ffs21 = 0, 0
    ffs12_lags, ffs12_percent_lags, ffs21_lags, ffs21_percent_lags = [], [], [], []

    if s1 == s2 and f1 == f2:
        ffs12 += 1
        ffs12_percent_lags.append(0.01)
        ffs12_lags.append(0.01)
    if f2 <= s1:
        fs21 += 1
    else:
        if s2 >= f1:
            fs12 += 1
        else:
            if s2 >= s1:
                if f2 >= f1:
                    ffs12 += 1
                    if get_delta_between_dates(f1, s1) != 0:
                        ffs12_percent_lags.append(get_delta_between_dates(s2, s1) / get_delta_between_dates(f1, s1))
                    else:
                        ffs12_percent_lags.append(0)
                    ffs12_lags.append(get_delta_between_dates(s2, s1))
                else:
                    ss12 += 1
                    if get_delta_between_dates(f1, s1) != 0:
                        ss12_percent_lags.append(get_delta_between_dates(s2, s1) / get_delta_between_dates(f1, s1))
                    else:
                        ss12_percent_lags.append(0)
                    ss12_lags.append(get_delta_between_dates(s2, s1))
            else:
                if f2 <= f1:
                    ffs21 += 1
                    if get_delta_between_dates(f2, s2) != 0:
                        ffs21_percent_lags.append(get_delta_between_dates(s1, s2) / get_delta_between_dates(f2, s2))
                    else:
                        ffs21_percent_lags.append(0)
                    ffs21_lags.append(get_delta_between_dates(s1, s2))
                else:
                    ss21 += 1
                    if get_delta_between_dates(f2, s2) != 0:
                        ss21_percent_lags.append(get_delta_between_dates(s1, s2) / get_delta_between_dates(f2, s2))
                    else:
                        ss21_percent_lags.append(0)
                    ss21_lags.append(get_delta_between_dates(s1, s2))
    return fs12, fs21, ss12, ss12_lags, ss12_percent_lags, ss21, ss21_lags, ss21_percent_lags, ffs12, ffs12_lags, ffs12_percent_lags, \
        ffs21, ffs21_lags, ffs21_percent_lags


def get_all_seq_statistic(history_data: pd.DataFrame, tasks_structure_df: pd.DataFrame, use_model_name: bool = False,
                          mapper: NameMapper = None) -> dict[int, list]:
    df_grouped = history_data.copy()

    if use_model_name:
        column_name = 'model_name'
    else:
        column_name = 'granular_smr_name'

    df_grouped = df_grouped.groupby('upper_works')[column_name].apply(list).reset_index(name="Works")
    works1, works2 = get_all_connections(tasks_structure_df, use_model_name, mapper)

    tasks_names = list(zip(works1['names'], works2['names']))
    tasks_ids = list(zip(works1['ids'], works2['ids']))

    predecessors_info_dict = {w_id: [] for w_id in tasks_structure_df['activity_id']}

    if len(tasks_names) != 0:
        for i in range(len(tasks_names)):
            w1, w2 = tasks_names[i]
            w1_id, w2_id = tasks_ids[i]

            if w1 != w2:
                fs12, fs21, ss12, ss21 = 0, 0, 0, 0
                ss12_lags, ss12_percent_lags, ss21_lags, ss21_percent_lags = [], [], [], []

                ffs12, ffs21 = 0, 0
                ffs12_lags, ffs12_percent_lags, ffs21_lags, ffs21_percent_lags = [], [], [], []

                for i, works_list in df_grouped.iterrows():
                    if w1 in works_list and w2 in works_list:
                        ind1 = history_data.loc[(history_data['upper_works'] == works_list['upper_works']) & (
                                    history_data['granular_smr_name'] == w1)]
                        ind2 = history_data.loc[(history_data['upper_works'] == works_list['upper_works']) & (
                                    history_data['granular_smr_name'] == w2)]

                        ind1_sorted = ind1.sort_values(by=['first_day', 'last_day']).reset_index(drop=True)
                        ind2_sorted = ind2.sort_values(by=['first_day', 'last_day']).reset_index(drop=True)

                        for l in range(min(len(ind1_sorted), len(ind2_sorted))):
                            s1, f1 = ind1_sorted.loc[l, 'first_day'], ind1_sorted.loc[l, 'last_day']

                            s2, f2 = ind2_sorted.loc[l, 'first_day'], ind2_sorted.loc[l, 'last_day']

                            if not any([type(x) == float for x in [s1, s2, f1, f2]]):
                                tasks_fs12, tasks_fs21, tasks_ss12, tasks_ss12_lags, tasks_ss12_percent_lags, tasks_ss21, tasks_ss21_lags, \
                                    tasks_ss21_percent_lags, tasks_ffs12, tasks_ffs12_lags, tasks_ffs12_percent_lags, tasks_ffs21, tasks_ffs21_lags, tasks_ffs21_percent_lags = gather_links_types_statistics(
                                    s1, f1, s2, f2)

                                fs12 += tasks_fs12
                                fs21 += tasks_fs21

                                ss12 += tasks_ss12
                                ss12_lags.extend(tasks_ss12_lags)
                                ss12_percent_lags.extend(tasks_ss12_percent_lags)
                                ss21 += tasks_ss21
                                ss21_lags.extend(tasks_ss21_lags)
                                ss21_percent_lags.extend(tasks_ss21_percent_lags)

                                ffs12 += tasks_ffs12
                                ffs12_lags.extend(tasks_ffs12_lags)
                                ffs12_percent_lags.extend(tasks_ffs12_percent_lags)
                                ffs21 += tasks_ffs21
                                ffs21_lags.extend(tasks_ffs21_lags)
                                ffs21_percent_lags.extend(tasks_ffs21_percent_lags)

                # Checking the mutual arrangement of tasks (follower - predecessor). We calculate which interposition is observed more often, correct the connections
                if fs12 + ffs12 + ss12 >= fs21 + ffs21 + ss21:
                    order_con = 1
                    fs = fs12
                    ffs = ffs12
                    ss = ss12
                else:
                    order_con = 2
                    fs = fs21
                    ffs = ffs21
                    ss = ss21

                if max([fs, ss, ffs]) != 0:
                    if fs > ss:
                        if ffs > 0:
                            if order_con == 1:
                                predecessors_info_dict[w2_id].append((w1_id, 'FFS',
                                                                      find_min_without_outliers(ffs12_percent_lags)))
                            else:
                                predecessors_info_dict[w1_id].append((w2_id, 'FFS',
                                                                      find_min_without_outliers(ffs21_percent_lags)))
                        else:
                            if order_con == 1:
                                predecessors_info_dict[w2_id].append((w1_id, 'FS', -1))
                            else:
                                predecessors_info_dict[w1_id].append((w2_id, 'FS', -1))
                    elif ss > ffs:
                        if order_con == 1:
                            predecessors_info_dict[w2_id].append((w1_id, 'SS',
                                                                  find_min_without_outliers(ss12_percent_lags)))
                        else:
                            predecessors_info_dict[w1_id].append((w2_id, 'SS',
                                                                  find_min_without_outliers(ss21_percent_lags)))
                    else:
                        if order_con == 1:
                            predecessors_info_dict[w2_id].append((w1_id, 'FFS',
                                                                  find_min_without_outliers(ffs12_percent_lags)))
                        else:
                            predecessors_info_dict[w1_id].append((w2_id, 'FFS',
                                                                  find_min_without_outliers(ffs21_percent_lags)))

    return predecessors_info_dict


def set_connections_info(structure_df: pd.DataFrame, history_data: pd.DataFrame, use_model_name: bool = False, mapper=None) \
        -> pd.DataFrame:
    """
    Restore tasks' connection based on history data

    :return: repaired DataFrame
    """
    tasks_df = structure_df.copy().set_index('activity_id')
    connections_dict = get_all_seq_statistic(history_data, structure_df, use_model_name, mapper)

    predecessors_ids_lst, predecessors_types_lst, predecessors_lags_lst = [], [], []

    for task_id, pred_info_lst in connections_dict.items():
        pred_ids_lst, pred_types_lst, pred_lags_lst = zip(*pred_info_lst)

        predecessors_ids_lst.append(','.join(map(str, pred_ids_lst)))
        predecessors_types_lst.append(','.join(map(str, pred_types_lst)))
        predecessors_lags_lst.append(','.join(map(str, pred_lags_lst)))

    # Convert strings to arrays
    tasks_df['predecessor_ids'] = predecessors_ids_lst
    tasks_df['connection_types'] = predecessors_types_lst
    tasks_df['lags'] = predecessors_lags_lst

    return tasks_df
