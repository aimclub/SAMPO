import pandas as pd
import ast
import json

from os import listdir
from os.path import isfile, join

from typing import Dict, List

import warnings
warnings.filterwarnings("ignore")


def get_full_resources_dict(workers_sets: List[Dict[str, float]]) -> Dict[str, int]:
    """
    Prepare resources dict
    :param workers_sets: list with info about assigned resources for tasks
    :return: dict with resources names as keys and resources ids as values
    """
    resources_names = []
    for workers_set in workers_sets:
        resources_names.extend(list(workers_set.keys()))
    resources_names = list(set(resources_names))
    resources_ids = range(len(resources_names))
    return dict(zip(resources_names, resources_ids))


# Main function for converting schedules to validation jsons
def schedule_to_json(schedule_uploading_mode='from_df', schedule_file_path='', schedule_df=None,
                     schedule_name='schedule') -> Dict:
    """
    Create dictionary with the JSON structure by given schedule and additional ksg info
    :param schedule_uploading_mode: one of two: 'from_df' (schedule_df needed)
    or 'from_file' (schedule_file_path needed)
    :param schedule_file_path: str path to the schedule .csv file
    :param schedule_df: pandas.Dataframe
    :param schedule_name: str
    :return: dict with JSON structure of tasks, assigned execution times and resources
    """
    if schedule_uploading_mode == 'from_file':
        df = pd.read_csv(schedule_file_path, sep=';')
    elif schedule_uploading_mode == 'from_df':
        df = schedule_df.copy()
        if not isinstance(schedule_df, pd.DataFrame):
            raise Exception("schedule_df attribute should have type pandas.DataFrame, got " +
                            str(type(schedule_df)) + " instead")
    else:
        raise Exception("unknown schedule uploading mode")

    df.loc[:, 'workers'] = [ast.literal_eval(x) for x in df.loc[:, 'workers']]
    resources_dict = get_full_resources_dict(df.loc[:, 'workers'])

    schedule_dict = {"plan_name": schedule_name, "activities": []}
    for i in df.index:
        if not ('start' in df.loc[i, 'task_name'] or 'finish' in df.loc[i, 'task_name']):
            activity_dict = {"activity_id": str(df.loc[i, 'task_id']),
                             "activity_name": df.loc[i, 'task_name'],
                             "start_date": str(df.loc[i, 'start']), "end_date": str(df.loc[i, 'finish']),
                             "volume": str(df.loc[i, 'volume']),
                             "measurement": str(df.loc[i, 'measurement']),
                             "labor_resources": [], "non_labor_resources": {},
                             "descendant_activities": df.loc[i, 'successors']}
            for worker in df.workers[i]:
                labor_info = {"labor_id": int(resources_dict[worker]), "labor_name": worker,
                              "volume": float(df.loc[i, 'workers'][worker]), "workers": []}
                worker_info = {"worker_id": 0, "contractor_id": 0, "contractor_name": 'Main contractor',
                               "working_periods": [{"start_datetime": str(df.loc[i, 'start']),
                                                    "end_datetime": str(df.loc[i, 'finish']),
                                                    "productivity": 1}]}
                labor_info["workers"].append(worker_info)
                activity_dict["labor_resources"].append(labor_info)
            schedule_dict["activities"].append(activity_dict)
    return schedule_dict


def get_info_path(ksg_block_name: str):
    if ksg_block_name == 'electroline':
        return '/brksg/messoyaha/electroline/works_info.csv'
    elif ksg_block_name == 'waterpipe':
        return '/brksg/new_port/R50/works_info.csv'
    else:
        raise Exception("unknown ksg name")


if __name__ == '__main__':
    RESOURCES_PATH = '../../../resources'

    schedules_files = [RESOURCES_PATH + '/schedules/' + f for f in listdir(RESOURCES_PATH + '/schedules/')
                       if isfile(join(RESOURCES_PATH + '/schedules/', f))]

    for schedule_file in schedules_files:
        schedule_title = schedule_file.split('/')[-1].split('.')[0]
        ksg_name = schedule_title.split('_')[0]

        print('Converting file: ', schedule_title + '.csv')
        schedule_json = schedule_to_json(schedule_uploading_mode='from_file', schedule_file_path=schedule_file,
                                         schedule_name=schedule_title)

        # Save file
        with open(RESOURCES_PATH + '/schedules/json/' + schedule_title + '.json', "w",
                  encoding='utf-8') as fp:
            json.dump(schedule_json, fp, ensure_ascii=False)
