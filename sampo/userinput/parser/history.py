import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from sampo.schemas.graph import EdgeType
from sampo.utilities.task_name import NameMapper


def get_all_connections(graph_df: pd.DataFrame, use_mapper: bool = False, mapper: NameMapper = None) \
        -> Tuple[dict[str, list], dict[str, list]]:

    task_name_column = 'activity_name'
    if 'granular_name' in graph_df:
        task_name_column = 'granular_name'

    num_tasks = len(graph_df)
    # Get the upper triangular indices to avoid duplicate pairs
    indices = np.triu_indices(num_tasks, k=1)
    works1_ids = graph_df['activity_id'].values[indices[0]]
    works1_names = graph_df[task_name_column].values[indices[0]]

    works2_ids = graph_df['activity_id'].values[indices[1]]
    works2_names = graph_df[task_name_column].values[indices[1]]

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


def get_all_seq_statistic(history_data, graph_df, use_model_name=False, mapper=None):
    df_grouped = history_data.copy()

    if use_model_name:
        column_name = 'model_name'
    else:
        column_name = 'granular_smr_name'

    df_grouped = df_grouped.groupby('upper_works')[column_name].apply(list).reset_index(name="Works")
    works1, works2 = get_all_connections(graph_df, use_model_name, mapper)

    # Declare structure with updated connections

    tasks_names = list(zip(works1['names'], works2['names']))
    tasks_ids = list(zip(works1['ids'], works2['ids']))

    predecessors_info_dict = {w_id: [] for w_id in graph_df['activity_id']}

    if len(tasks_names) != 0:
        for i in range(len(tasks_names)):
            w1, w2 = tasks_names[i]
            w1_id, w2_id = tasks_ids[i]

            if w1 != w2:
                fs12, fs21, ss12, ss21 = 0, 0, 0, 0
                ss12_lags, ss12_percent_lags, ss21_lags, ss21_percent_lags = [], [], [], []

                count = 0

                ffs12, ffs21 = 0, 0
                ffs12_lags, ffs12_percent_lags, ffs21_lags, ffs21_percent_lags = [], [], [], []

                for i, work_list in df_grouped.iterrows():
                    # Looking to see if this pair of works occurred within the same site in the historical data
                    if w1 in work_list['Works'] and w2 in work_list['Works']:
                        ind1 = history_data.loc[(history_data['upper_works'] == work_list['upper_works']) &
                                                (history_data['granular_smr_name'] == w1)]
                        ind2 = history_data.loc[(history_data['upper_works'] == work_list['upper_works']) &
                                                (history_data['granular_smr_name'] == w2)]

                        ind1_sorted = ind1.sort_values(by=['first_day', 'last_day']).reset_index(drop=True)
                        ind2_sorted = ind2.sort_values(by=['first_day', 'last_day']).reset_index(drop=True)

                        for l in range(min(len(ind1_sorted), len(ind2_sorted))):
                            s1, f1 = ind1_sorted.loc[l, 'first_day'], ind1_sorted.loc[l, 'last_day']

                            s2, f2 = ind2_sorted.loc[l, 'first_day'], ind2_sorted.loc[l, 'last_day']

                            if not any([type(x) == float for x in [s1, s2, f1, f2]]):
                                tasks_fs12, tasks_fs21, tasks_ss12, tasks_ss12_lags, tasks_ss12_percent_lags, tasks_ss21, tasks_ss21_lags, \
                                    tasks_ss21_percent_lags, tasks_ffs12, tasks_ffs12_lags, tasks_ffs12_percent_lags, tasks_ffs21, tasks_ffs21_lags, tasks_ffs21_percent_lags = gather_links_types_statistics(
                                    s1, f1, s2, f2)

                                count += 1

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
                                predecessors_info_dict[w2_id].append([w1_id, 'FFS',
                                                                      find_min_without_outliers(ffs12_percent_lags),
                                                                      count])
                            else:
                                predecessors_info_dict[w1_id].append([w2_id, 'FFS',
                                                                      find_min_without_outliers(ffs21_percent_lags),
                                                                      count])
                        else:
                            if order_con == 1:
                                predecessors_info_dict[w2_id].append([w1_id, 'FS', -1, count])
                            else:
                                predecessors_info_dict[w1_id].append([w2_id, 'FS', -1, count])
                    elif ss > ffs:
                        if order_con == 1:
                                predecessors_info_dict[w2_id].append([w1_id, 'SS',
                                                                      find_min_without_outliers(ss12_percent_lags),
                                                                      count])
                        else:
                            predecessors_info_dict[w1_id].append([w2_id, 'SS',
                                                                  find_min_without_outliers(ss21_percent_lags), count])
                    else:
                        if order_con == 1:
                                predecessors_info_dict[w2_id].append([w1_id, 'FFS',
                                                                      find_min_without_outliers(ffs12_percent_lags),
                                                                      count])
                        else:
                            predecessors_info_dict[w1_id].append([w2_id, 'FFS',
                                                                  find_min_without_outliers(ffs21_percent_lags), count])

    return predecessors_info_dict


def set_connections_info(graph_df: pd.DataFrame,
                         history_data: pd.DataFrame,
                         use_model_name: bool = False,
                         mapper=None,
                         change_connections_info: bool = False,
                         expert_connections_info: bool = False) \
        -> pd.DataFrame:
    """
    Restore tasks' connection based on history data

    :param: change_connections_info - whether existing connections should be modified based on connection history data
    :param: expert_connections_info - whether existing connections should not be modified based on connection history data
    :return: repaired DataFrame
    """
    tasks_df = graph_df.copy().set_index('activity_id', drop=False)
    connections_dict = get_all_seq_statistic(history_data, graph_df, use_model_name, mapper)
    # connections_dict = {'25809398': [], '25809830': [], '25809831': [['25809830', 'FFS', 0.01, 185], ['25809856', 'FS', -1, 3]], '25809833': [['25809830', 'FFS', 0.01, 276], ['25809831', 'FFS', 0.01, 1646]], '25813507': [['25809830', 'FFS', 0.01, 60], ['25809831', 'FFS', 0.01, 131], ['25809833', 'FFS', 0.01, 423]], '25809836': [['25809830', 'FFS', 0.01, 113], ['25809831', 'FFS', 0.01, 907], ['25809833', 'FFS', 0.01, 1310], ['25813507', 'FFS', 0.01, 278]], '25809832': [['25809830', 'FS', -1, 3], ['25809833', 'FS', -1, 3], ['25809836', 'FFS', 0.01, 5], ['25809839', 'SS', 0.05, 6], ['25809852', 'FS', -1, 2], ['25809854', 'FS', -1, 2]], '25809837': [['25809831', 'FS', -1, 2], ['25809836', 'SS', 0.75, 2], ['25809835', 'FS', -1, 2]], '25809838': [['25809830', 'FS', -1, 3], ['25809831', 'FFS', 0.01, 259], ['25809833', 'FFS', 0.01, 379], ['25813507', 'FFS', 0.01, 111], ['25809836', 'FFS', 0.01, 500]], '25809839': [['25809830', 'FFS', 0.52, 25], ['25809831', 'FFS', 0.01, 345], ['25809833', 'FFS', 0.01, 419], ['25813507', 'FFS', 0.33, 61], ['25809836', 'FFS', 0.01, 497], ['25809838', 'FFS', 0.01, 372], ['25809835', 'FS', -1, 2], ['25809847', 'FFS', 0.27, 10], ['25809850', 'FS', -1, 1], ['25809852', 'FFS', 0.05, 97], ['25809848', 'FFS', 0.27, 10], ['25809853', 'FS', -1, 1], ['25809854', 'FFS', 0.05, 97]], '25809834': [['25809830', 'FS', -1, 5], ['25809831', 'FS', -1, 6], ['25809833', 'FS', -1, 7], ['25809836', 'FFS', 0.38, 13], ['25809838', 'FFS', 0.5, 8], ['25809839', 'SS', 0.68, 4], ['25809835', 'FS', -1, 18], ['25809840', 'FS', -1, 18], ['25809841', 'FFS', 0.86, 12]], '25809835': [['25809831', 'FFS', 0.04, 6], ['25809833', 'FFS', 0.04, 6], ['25809836', 'SS', 0.19, 8], ['25809838', 'SS', 0.5, 5], ['25809841', 'SS', 0.76, 1], ['25809852', 'FS', -1, 2], ['25809854', 'FS', -1, 2]], '25809842': [['25809830', 'FFS', 0.01, 6], ['25809831', 'FFS', 0.01, 2], ['25809833', 'FFS', 0.01, 8], ['25813507', 'FFS', 0.25, 1], ['25809836', 'FFS', 0.01, 9], ['25809832', 'FFS', 0.01, 1], ['25809839', 'FFS', 0.01, 2], ['25809840', 'SS', 0.79, 1], ['25809841', 'FS', -1, 11]], '25809843': [['25809832', 'FFS', 0.01, 23], ['25809839', 'FS', -1, 1], ['25809842', 'FFS', 0.06, 1], ['25809841', 'FS', -1, 4]], '25809844': [['25809832', 'FFS', 0.01, 4], ['25809843', 'SS', 0.03, 4], ['25809841', 'FS', -1, 2]], '25809840': [['25809830', 'FS', -1, 2], ['25809831', 'FFS', 0.01, 40], ['25809833', 'FFS', 0.01, 43], ['25809836', 'FFS', 0.17, 45], ['25809838', 'FS', -1, 8], ['25809839', 'FS', -1, 13], ['25809835', 'FFS', 0.01, 24], ['25809841', 'FS', -1, 19], ['25809847', 'FS', -1, 7], ['25809850', 'FFS', 0.12, 1], ['25809848', 'FS', -1, 7], ['25809853', 'FFS', 0.12, 1]], '25809841': [['25809830', 'FFS', 0.01, 202], ['25809831', 'FFS', 0.01, 334], ['25809833', 'FFS', 0.01, 446], ['25813507', 'FFS', 0.01, 105], ['25809836', 'FFS', 0.01, 358], ['25809832', 'SS', 0.32, 7], ['25809838', 'FFS', 0.01, 103], ['25809839', 'FFS', 0.01, 190], ['25809847', 'FS', -1, 17], ['25809850', 'FS', -1, 2], ['25809852', 'FS', -1, 2], ['25809848', 'FS', -1, 17], ['25809853', 'FS', -1, 2], ['25809854', 'FS', -1, 2], ['25809856', 'FS', -1, 1]], '25809399': [], '25809400': [], '25809847': [['25809830', 'FS', -1, 1], ['25809831', 'FFS', 0.01, 81], ['25809833', 'FFS', 0.01, 85], ['25809836', 'FFS', 0.01, 133], ['25809838', 'FFS', 0.01, 54], ['25809834', 'FS', -1, 2], ['25809850', 'FFS', 0.03, 68], ['25809853', 'FFS', 0.03, 68], ['25809856', 'FS', -1, 2]], '25809850': [['25809831', 'FS', -1, 3], ['25809833', 'FS', -1, 3], ['25809836', 'FFS', 0.67, 8], ['25809838', 'FFS', 0.03, 44], ['25809856', 'FS', -1, 3]], '25809852': [['25809831', 'FFS', 0.68, 3], ['25809833', 'FFS', 0.68, 3], ['25809836', 'FFS', 0.46, 6], ['25809838', 'SS', 0.01, 2], ['25809847', 'FFS', 0.01, 9], ['25809850', 'FFS', 0.01, 265], ['25809857', 'FS', -1, 1], ['25809855', 'FFS', 0.16, 20]], '25809848': [['25809830', 'FS', -1, 1], ['25809831', 'FFS', 0.01, 81], ['25809833', 'FFS', 0.01, 85], ['25809836', 'FFS', 0.01, 133], ['25809838', 'FFS', 0.01, 54], ['25809834', 'FS', -1, 2], ['25809850', 'FFS', 0.01, 68], ['25809852', 'FFS', 0.01, 9], ['25809853', 'FFS', 0.03, 68], ['25809856', 'FS', -1, 2]], '25809853': [['25809831', 'FS', -1, 3], ['25809833', 'FS', -1, 3], ['25809836', 'FFS', 0.67, 8], ['25809838', 'FFS', 0.03, 44], ['25809852', 'FFS', 0.01, 265], ['25809856', 'FS', -1, 3]], '25809854': [['25809831', 'FFS', 0.68, 3], ['25809833', 'FFS', 0.68, 3], ['25809836', 'FFS', 0.46, 6], ['25809838', 'SS', 0.01, 2], ['25809847', 'FFS', 0.01, 9], ['25809850', 'FFS', 0.01, 265], ['25809848', 'FFS', 0.01, 9], ['25809853', 'FFS', 0.01, 265], ['25809857', 'FS', -1, 1], ['25809855', 'FFS', 0.16, 20]], '25809856': [['25809833', 'FS', -1, 43], ['25813507', 'FFS', 0.91, 33], ['25809836', 'FFS', 0.47, 41], ['25809838', 'FFS', 0.01, 7], ['25809839', 'FFS', 0.01, 12], ['25809835', 'FS', -1, 1]], '25809401': [], '25809857': [['25809855', 'FS', -1, 1]], '25809858': [['25809833', 'FS', -1, 1], ['25809836', 'FS', -1, 1], ['25809838', 'FS', -1, 1], ['25809839', 'FFS', 0.01, 56], ['25809841', 'SS', 0.36, 4], ['25809847', 'FFS', 0.01, 17], ['25809850', 'FFS', 0.93, 91], ['25809852', 'FS', -1, 89], ['25809848', 'FFS', 0.01, 17], ['25809853', 'FFS', 0.93, 91], ['25809854', 'FS', -1, 89], ['25809856', 'SS', 0.64, 1], ['25809857', 'FS', -1, 1]], '25809855': [['25809850', 'FFS', 0.12, 4], ['25809853', 'FFS', 0.12, 4]]}

    predecessors_ids_lst, predecessors_types_lst, predecessors_lags_lst, predecessors_counts_lst = [], [], [], []

    for task_id, pred_info_lst in connections_dict.items():
        if len(pred_info_lst) > 0:
            pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst = map(list, zip(*pred_info_lst))
        else:
            pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst = ['-1'], ['-1'], [-1], [0]

        if str(task_id) in tasks_df.index:
            if tasks_df.loc[str(task_id), 'predecessor_ids'] != ['-1']:
                if expert_connections_info:
                    predecessors_ids_lst.append(tasks_df.loc[str(task_id), 'predecessor_ids'])
                    predecessors_types_lst.append(tasks_df.loc[str(task_id), 'connection_types'])
                    predecessors_lags_lst.append(tasks_df.loc[str(task_id), 'lags'])
                    predecessors_counts_lst.append(pred_counts_lst)
                    continue
                if change_connections_info:
                    predecessors_ids_lst.append(tasks_df.loc[str(task_id), 'predecessor_ids'])
                else:
                    predecessors_ids_lst.append(pred_ids_lst)
            else:
                predecessors_ids_lst.append(pred_ids_lst)
        else:
            predecessors_ids_lst.append(pred_ids_lst)

        predecessors_types_lst.append(pred_types_lst)
        predecessors_lags_lst.append(pred_lags_lst)
        predecessors_counts_lst.append(pred_counts_lst)
        while len(predecessors_types_lst[-1]) != len(predecessors_ids_lst[-1]):
            predecessors_types_lst[-1].append('FS')
            predecessors_lags_lst[-1].append(-1)
            predecessors_counts_lst[-1].append(0)

    # Convert strings to arrays
    tasks_df['predecessor_ids'] = predecessors_ids_lst
    tasks_df['connection_types'] = predecessors_types_lst
    tasks_df['lags'] = predecessors_lags_lst
    tasks_df['counts'] = predecessors_counts_lst

    tasks_df['connection_types'] = tasks_df['connection_types'].apply(
        lambda x: [EdgeType(elem) if elem != '-1' else EdgeType.FinishStart for elem in x]
    )

    return tasks_df
