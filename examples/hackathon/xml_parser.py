from uuid import uuid4

import pandas as pd

from sampo.schemas.resources import Worker
from sampo.schemas.contractor import Contractor

import xml.etree.ElementTree as ET

from datetime import timedelta, date
from business_calendar import Calendar, MO, TU, WE, TH, FR, SA, SU

tag_prefix = '{http://schemas.microsoft.com/project}'

FS_lag = '0.0'

work_req_dict = {
    'Аналитика': 'Аналитик',
    'Разработка': 'Разработчик',
    'Тестирование': 'Тестировщик'
}

week_days_by_id = {
    '1': SU,
    '2': MO,
    '3': TU,
    '4': WE,
    '5': TH,
    '6': FR,
    '7': SA}


# Helper functions
def get_resources_info(filepath: str) -> list[dict]:
    # Parse XML file with the project's data
    project_tree = ET.parse(filepath)
    project_root = project_tree.getroot()

    workers_lst = []
    project_resources = project_root.find(tag_prefix + 'Resources').\
        findall(tag_prefix + 'Resource')

    # Resources info preparation
    for res_info in project_resources:

        res_id = res_info.find(tag_prefix + 'UID').text

        res_name = res_info.find(tag_prefix + 'Name').text
        res_cost = int(res_name.split('(')[-1].split('руб/час)')[0]) if '(' in res_name else 0

        attributes = res_info.findall(tag_prefix + 'ExtendedAttribute')
        res_skills = set()
        for attr in attributes:
            if attr.find(tag_prefix + 'FieldID').text == '205521131':
                res_skills.add(attr.find(tag_prefix + 'Value').text)

        workers_lst.append({
            'name': res_name,
            'count': 1,
            'cost_one_unit': res_cost,
            'skills': res_skills
        })

    return workers_lst


def get_calendar_info(path_to_input_xml):
    # Parse XML file with the project's data
    project_tree = ET.parse(path_to_input_xml)
    project_root = project_tree.getroot()

    project_calendars = project_root.find(tag_prefix + 'Calendars').findall(tag_prefix + 'Calendar')

    # Get info about current project calendar
    calendar_id = project_root.find(tag_prefix + 'CalendarUID').text

    main_calendar = None

    for calendar in project_calendars:
        if calendar.find(tag_prefix + 'UID').text == calendar_id:
            main_calendar = calendar

    return main_calendar


def generate_dates(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def get_works_info(filepath: str) -> pd.DataFrame:
    # Parse XML file with the project's data
    project_tree = ET.parse(filepath)
    project_root = project_tree.getroot()

    # Prepare structure for the input data after parsing XML
    input_data = {
        'activity_id': [],
        'activity_name': [],
        'wbs_id': [],
        'wbs_level': [],
        'granular_name': [],
        'volume': [],
        'measurement': [],
        'predecessor_ids': [],
        'connection_types': [],
        'lags': []
    }

    # For hierarchical structure
    project_wbs_numbers = {}
    project_wbs = {'task_id': [],
                   'predecessor_id': []}
    project_wbs_levels = {}

    executable_successors_dict = {}
    successors_dict = {}

    project_tasks = project_root.find(tag_prefix + 'Tasks').\
        findall(tag_prefix + 'Task')

    # Tasks info preparation
    for task_info in project_tasks:

        task_id = task_info.find(tag_prefix + 'UID').text
        input_data['activity_id'].append(task_id)

        task_name = task_info.find(tag_prefix + 'Name').text
        input_data['activity_name'].append(task_name)
        input_data['granular_name'].append(task_name.split(' ')[0])

        volume_str = task_info.find(tag_prefix + 'Duration').text
        input_data['volume'].append(int(volume_str.split('PT')[1].split('H')[0]))  # in hours
        input_data['measurement'].append('hours')

        task_wbs_number = task_info.find(tag_prefix + 'OutlineNumber').text
        project_wbs_numbers[task_wbs_number] = task_id  # add task wbs number to the set of all wbs_numbers
        input_data['wbs_id'].append(task_wbs_number)

        task_wbs_level = int(task_info.find(tag_prefix + 'OutlineLevel').text)
        if task_wbs_level not in project_wbs_levels:
            project_wbs_levels[task_wbs_level] = set()
        project_wbs_levels[task_wbs_level].add(task_id)
        input_data['wbs_level'].append(task_wbs_level)

        predecessors_info = task_info.findall(tag_prefix + 'PredecessorLink')
        predecessor_ids = []
        connection_types = []
        lags = []

        for predecessor in predecessors_info:
            predecessor_ids.append(predecessor.find(tag_prefix + 'PredecessorUID').text)
            connection_types.append('FS')
            if int(predecessor.find(tag_prefix + 'LinkLag').text) == 0:
                lags.append(FS_lag)
            else:
                lags.append(str(int(predecessor.find(tag_prefix + 'LinkLag').text)))

        input_data['predecessor_ids'].append(predecessor_ids)
        input_data['connection_types'].append(connection_types)
        input_data['lags'].append(lags)

    project_df = pd.DataFrame(data=input_data)

    # Get info about tasks predecessors in project hierarchy
    for i in range(len(project_df)):
        wbs_number = '.'.join(project_df.loc[i, 'wbs_id'].split('.')[:-1])
        if wbs_number in project_wbs_numbers:
            task_id = project_df.loc[i, 'activity_id']
            predecessor_id = project_wbs_numbers[wbs_number]
            project_wbs['task_id'].append(task_id)
            project_wbs['predecessor_id'].append(predecessor_id)
            if predecessor_id not in successors_dict:
                successors_dict[predecessor_id] = set()
            successors_dict[predecessor_id].add(task_id)

    # Get information about which tasks are executable
    is_executable_lst = []
    executable_dict = {}

    for i in range(len(project_df)):
        if project_df.loc[i, 'activity_id'] in project_wbs['predecessor_id']:
            is_executable_lst.append(0)
            executable_dict[project_df.loc[i, 'activity_id']] = 0
        else:
            is_executable_lst.append(1)
            executable_dict[project_df.loc[i, 'activity_id']] = 1

    project_df['is_executable'] = is_executable_lst

    # Get info about tasks predecessors in project hierarchy
    for i in range(len(project_df)):
        if project_df.loc[i, 'is_executable'] == 0:
            task_wbs_id = project_df.loc[i, 'wbs_id']
            task_id = project_df.loc[i, 'activity_id']
            for j in range(len(project_df)):
                if project_df.loc[j, 'activity_id'] != task_id and \
                        project_df.loc[j, 'wbs_id'].startswith(task_wbs_id) and \
                        project_df.loc[j, 'is_executable'] == 1:
                    if task_id not in executable_successors_dict:
                        executable_successors_dict[task_id] = set()
                    executable_successors_dict[task_id].add(project_df.loc[j, 'activity_id'])

    project_df = project_df.set_index('activity_id')

    new_vertices = []
    for wbs_lvl in range(1, max(project_wbs_levels.keys()) + 1):
        for task_id in project_wbs_levels[wbs_lvl]:
            if project_df.loc[task_id, 'is_executable'] == 0:
                # Create info about start vertex for the sub-project
                start_vertex = {
                    'activity_id': task_id + '0000',
                    'activity_name': 'Начало работ ' + project_df.loc[task_id, 'activity_name'],
                    'wbs_id': project_df.loc[task_id, 'wbs_id'],
                    'wbs_level': project_df.loc[task_id, 'wbs_level'],
                    'granular_name': 'Начало работ',
                    'volume': 0,
                    'measurement': 'hours',
                    'predecessor_ids': project_df.loc[task_id, 'predecessor_ids'],
                    'connection_types': project_df.loc[task_id, 'connection_types'],
                    'lags': project_df.loc[task_id, 'lags'],
                    'is_executable': 0,
                    'is_service': 1
                }
                new_vertices.append(start_vertex)

                # Create info about finish vertex for the sub-project
                finish_vertex = {
                    'activity_id': task_id + '1111',
                    'activity_name': 'Окончание работ ' + project_df.loc[task_id, 'activity_name'],
                    'wbs_id': project_df.loc[task_id, 'wbs_id'],
                    'wbs_level': project_df.loc[task_id, 'wbs_level'],
                    'granular_name': 'Окончание работ',
                    'volume': 0,
                    'measurement': 'hours',
                    'predecessor_ids': list(successors_dict[task_id]),
                    'connection_types': ['FS'] * len(successors_dict[task_id]),
                    'lags': [FS_lag] * len(successors_dict[task_id]),
                    'is_executable': 0,
                    'is_service': 1
                }
                new_vertices.append(finish_vertex)

                # Add link to the start vertex to all inner tasks of the given sub-project
                for inner_task_id in successors_dict[task_id]:
                    project_df.loc[inner_task_id, 'predecessor_ids'].append(task_id + '0000')
                    project_df.loc[inner_task_id, 'connection_types'].append('FS')
                    project_df.loc[inner_task_id, 'lags'].append(FS_lag)

    project_df['is_service'] = [0] * len(project_df)
    service_tasks_df = pd.DataFrame(data=new_vertices)
    project_df = project_df.reset_index()
    project_df = pd.concat([project_df, service_tasks_df])
    project_df = project_df.reset_index(drop=True)
    project_df = project_df.set_index('activity_id')

    # Fix links from non-executable tasks
    new_predecessors_lst = []
    for i in project_df.index:
        upd_predecessors_lst = []
        for predecessor_id in project_df.loc[i, 'predecessor_ids']:
            if predecessor_id in project_df.index and project_df.loc[predecessor_id, 'is_executable'] == 0 and \
                    project_df.loc[predecessor_id, 'is_service'] == 0:
                upd_predecessors_lst.append(predecessor_id + '1111')  # link to the finish of the task
            else:
                upd_predecessors_lst.append(predecessor_id)
        new_predecessors_lst.append(upd_predecessors_lst)

    project_df = project_df.reset_index()
    project_df['predecessor_ids'] = new_predecessors_lst

    # Prepare dataframe with works info for scheduling
    df_for_scheduling = project_df[(project_df['is_executable'] == 1) | (project_df['is_service'] == 1)]
    df_for_scheduling = df_for_scheduling.reset_index(drop=True)
    df_for_scheduling = df_for_scheduling[
        ['activity_id', 'activity_name', 'granular_name', 'volume', 'measurement', 'predecessor_ids',
         'connection_types', 'lags', 'is_service']]

    # Fix lists representation
    new_predecessors_ids = [','.join(preds_lst) for preds_lst in df_for_scheduling['predecessor_ids']]
    new_connection_types = [','.join(preds_lst) for preds_lst in df_for_scheduling['connection_types']]
    new_lags = [','.join(preds_lst) for preds_lst in df_for_scheduling['lags']]

    df_for_scheduling['predecessor_ids'] = new_predecessors_ids
    df_for_scheduling['connection_types'] = new_connection_types
    df_for_scheduling['lags'] = new_lags

    # Add information about requirements
    workers_lst = get_resources_info(filepath)

    min_req = []
    max_req = []
    volume = []

    for i in range(len(df_for_scheduling)):
        min_reqs = {}
        max_reqs = {}
        volumes = {}
        for worker in workers_lst:
            if df_for_scheduling.loc[i, 'granular_name'] in work_req_dict and \
                    work_req_dict[df_for_scheduling.loc[i, 'granular_name']] in worker['skills']:
                min_reqs[worker['name']] = 0
                max_reqs[worker['name']] = 1
                volumes[worker['name']] = df_for_scheduling.loc[i, 'volume']
        min_req.append(min_reqs)
        max_req.append(max_reqs)
        volume.append(volumes)

    df_for_scheduling['min_req'] = min_req
    df_for_scheduling['max_req'] = max_req
    df_for_scheduling['req_volume'] = volume

    return df_for_scheduling


def get_contractors_info(filepath:str) -> list[Contractor]:
    workers_dict = {}
    workers_lst = get_resources_info(filepath)

    for worker in workers_lst:
        workers_dict[worker['name']] = Worker(
            id=str(uuid4()),
            name=worker['name'],
            count=worker['count'],
            cost_one_unit=worker['cost_one_unit']
        )

    contractors = [
        Contractor(id=str(uuid4()),
                   name="Main project contractor",
                   workers=workers_dict)
    ]

    return contractors


def get_project_calendar(path_to_input_xml: str) -> Calendar:
    main_calendar = get_calendar_info(path_to_input_xml)

    # Get information about working days and holidays from WeekDays tag
    week_days = main_calendar.find(tag_prefix + 'WeekDays').findall(tag_prefix + 'WeekDay')

    working_days = []
    holidays_lst = []

    for week_day in week_days:
        if week_day.find(tag_prefix + 'DayWorking').text == '1':
            working_days.append(week_days_by_id[week_day.find(tag_prefix + 'DayType').text])
        elif week_day.find(tag_prefix + 'DayType').text == '0':
            time_period = week_day.find(tag_prefix + 'TimePeriod')
            if time_period is not None:
                start_date_lst = time_period.find(tag_prefix + 'FromDate').text.split('T')[0].split('-')
                start_date = date(*[int(x) for x in start_date_lst])
                end_date_lst = time_period.find(tag_prefix + 'ToDate').text.split('T')[0].split('-')
                end_date = date(*[int(x) for x in end_date_lst])
                holidays_lst.extend([x.strftime("%Y-%m-%d") for x in list(generate_dates(start_date, end_date))])
    holidays = set(holidays_lst)

    # Get information about exceptions from Exceptions tag
    exceptions = main_calendar.find(tag_prefix + 'Exceptions').findall(tag_prefix + 'Exception')

    for exception in exceptions:
        time_period = exception.find(tag_prefix + 'TimePeriod')
        is_working_day = exception.find(tag_prefix + 'DayWorking').text

        if time_period is not None:
            start_date_lst = time_period.find(tag_prefix + 'FromDate').text.split('T')[0].split('-')
            start_date = date(*[int(x) for x in start_date_lst])
            end_date_lst = time_period.find(tag_prefix + 'ToDate').text.split('T')[0].split('-')
            end_date = date(*[int(x) for x in end_date_lst])

            if is_working_day == '0':
                for x in list(generate_dates(start_date, end_date)):
                    holidays.add(x.strftime("%Y-%m-%d"))
            else:
                for x in list(generate_dates(start_date, end_date)):
                    if x.strftime("%Y-%m-%d") in holidays:
                        holidays.remove(x.strftime("%Y-%m-%d"))

    return Calendar(workdays=working_days, holidays=list(holidays))
