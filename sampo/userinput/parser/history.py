import datetime
import math
from typing import Tuple

import numpy as np
import pandas as pd

from sampo.schemas.graph import EdgeType
from sampo.utilities.name_mapper import NameMapper


def get_all_connections(graph_df: pd.DataFrame,
                        use_mapper: bool = False,
                        mapper: NameMapper | None = None) \
        -> Tuple[dict[str, list], dict[str, list]]:

    task_name_column = 'model_name'

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


def add_granular_name_if_absent(row) -> str:
    model_name = row['model_name']
    if 'granular_name' not in model_name:
        model_name['granular_name'] = row['work_name']
    return model_name


def get_all_seq_statistic(history_data: pd.DataFrame,
                          graph_df: pd.DataFrame,
                          use_model_name: bool = False,
                          mapper: NameMapper | None = None):
    if 'model_name' not in history_data.columns:
        history_data['model_name'] = [{} for _ in range(len(history_data))]

    if len(history_data) > 0:
        history_data['model_name'] = history_data.apply(add_granular_name_if_absent, axis=1)
    # [{'granular_name': activity_name} for activity_name in history_data['work_name']]
    column_name = 'model_name'

    df_grouped = history_data.copy()

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
                                                (history_data[column_name] == w1)]
                        ind2 = history_data.loc[(history_data['upper_works'] == work_list['upper_works']) &
                                                (history_data[column_name] == w2)]

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
                                predecessors_info_dict[w2_id].append([w1_id, 'FS', 0.0, count])
                            else:
                                predecessors_info_dict[w1_id].append([w2_id, 'FS', 0.0, count])
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
                         mapper: NameMapper | None = None,
                         change_connections_info: bool = False,
                         all_connections: bool = False,
                         id2ind: dict[str, int] = None) \
        -> pd.DataFrame:
    """
    Restore tasks' connection based on history data

    :param: change_connections_info - whether existing connections' information should be modified based on history data
    :param: expert_connections_info - whether existing connections should not be modified based on connection history data
    :return: repaired DataFrame
    """

    def update_connections(pred_ids_lst_tmp, pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst):
        """Updates existing connections with new data."""
        for i, pred_id in enumerate(pred_ids_lst_tmp):
            if pred_id in pred_ids_lst:
                idx = pred_ids_lst.index(pred_id)
                pred_types_lst_tmp[i] = pred_types_lst[idx]
                pred_lags_lst_tmp[i] = pred_lags_lst[idx]
                pred_counts_lst_tmp[i] = pred_counts_lst[idx]
        return pred_ids_lst_tmp, pred_types_lst_tmp, pred_lags_lst_tmp, pred_counts_lst_tmp

    def append_new_connections(pred_ids_lst_tmp, pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst):
        """Appends new connections to the existing lists."""
        for i, pred_id in enumerate(pred_ids_lst):
            if pred_id not in pred_ids_lst_tmp and pred_id != '-1' and pred_id in all_works:
                pred_ids_lst_tmp.append(pred_id)
                pred_types_lst_tmp.append(pred_types_lst[i])
                pred_lags_lst_tmp.append(pred_lags_lst[i])
                pred_counts_lst_tmp.append(pred_counts_lst[i])
        return pred_ids_lst_tmp, pred_types_lst_tmp, pred_lags_lst_tmp, pred_counts_lst_tmp

    tasks_df = graph_df.copy()

    INF = np.iinfo(np.int64).max

    # | ------ no changes ------- |
    if not change_connections_info and all_connections:
        tasks_df['counts'] = [[INF] * len(tasks_df['predecessor_ids'][i]) for i in range(tasks_df.shape[0])]

        tasks_df['connection_types'] = tasks_df['connection_types'].apply(
            lambda x: [EdgeType(elem) if elem != '-1' else EdgeType.FinishStart for elem in x]
        )
        return tasks_df

    # | ----------- for cache data ----------- |
    connections_dict = get_all_seq_statistic(history_data, graph_df, use_model_name, mapper)
    # with open('dormitory.json', 'w') as f:
    #     json.dump(connections_dict, f)

    all_works = tasks_df['activity_id'].values

    for task_id, pred_info_lst in connections_dict.items():
        if str(task_id) not in all_works:
            continue

        row = tasks_df.loc[id2ind[str(task_id)]]

        # | ------ change links and info about them ------ |
        # by default, the links and their information change
        if len(pred_info_lst) > 0:
            pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst = map(list, zip(*pred_info_lst))
        else:
            pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst = ['-1'], ['-1'], [1], [INF]

        if not all_connections:
            pred_ids_lst_tmp, pred_types_lst_tmp, pred_lags_lst_tmp, pred_counts_lst_tmp = (
                row['predecessor_ids'].copy(), row['connection_types'].copy(), row['lags'].copy(), [INF] * len(row['lags']))
            if change_connections_info:
                pred_ids_lst_tmp, pred_types_lst_tmp, pred_lags_lst_tmp, pred_counts_lst_tmp = (
                    update_connections(pred_ids_lst_tmp, pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst))
            pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst = (
                append_new_connections(pred_ids_lst_tmp, pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst))
        else:
            if row['predecessor_ids'] != ['-1']:
                # if 'lags' is unknown, thus 'connection_type' is also unknown
                pred_info_lst = [[pred_types_lst[i], pred_lags_lst[i], pred_counts_lst[i]]
                                 for i in range(len(pred_ids_lst)) if pred_ids_lst[i] in row['predecessor_ids']]
                pred_ids_lst = row['predecessor_ids']
                if len(pred_info_lst) > 0:
                    pred_types_lst, pred_lags_lst, pred_counts_lst = map(list, zip(*pred_info_lst))
                else:
                    pred_types_lst, pred_lags_lst, pred_counts_lst = ['-1'], [1], [INF]

        while len(pred_ids_lst) != len(pred_types_lst):
            pred_types_lst.append('FS')
            pred_lags_lst.append(1)
            pred_counts_lst.append(0)
        for col, val in zip(['predecessor_ids', 'connection_types', 'lags', 'counts'], [pred_ids_lst, pred_types_lst, pred_lags_lst, pred_counts_lst]):
            tasks_df.at[id2ind[str(task_id)], col] = val

    tasks_df['connection_types'] = tasks_df['connection_types'].apply(
        lambda x: [EdgeType(elem) if elem != '-1' else EdgeType.FinishStart for elem in x]
    )

    return tasks_df
