from uuid import uuid4

import pandas as pd
from ast import literal_eval

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
    'Тестирование': 'Тестировщик',
    'Общая': 'Разработчик'
}

works_with_spec = {'Аналитика', 'Разработка', 'Тестирование'}

week_days_by_id = {
    '1': 6,
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5}


# Helper functions
def get_resources_info(filepath: str) -> list[dict]:
    # Parse XML file with the project's data
    parser = ET.XMLParser(encoding="utf-8")
    project_tree = ET.parse(filepath, parser=parser)
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
            'id': res_id,
            'name': res_name,
            'count': 1,
            'cost_one_unit': res_cost,
            'skills': res_skills
        })

    return workers_lst


def get_calendar_info(filepath):
    # Parse XML file with the project's data
    parser = ET.XMLParser(encoding="utf-8")
    project_tree = ET.parse(filepath, parser=parser)
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


def get_datetime_string(project_start_date, start_hours, end_hours):
    start_date = project_start_date + timedelta(days=start_hours // 8)
    add_hours = start_hours % 8 + 9
    if start_hours % 8 > 3:
        add_hours = add_hours + 1
    start_datetime = start_date.strftime("%Y-%m-%d") + 'T' + str(add_hours) + ':00:00'

    end_date = project_start_date + timedelta(days=end_hours // 8)
    add_end_hours = end_hours % 8 + 9
    if (end_hours % 8) == 0:
        end_date = end_date - timedelta(days=1)
        add_end_hours = 17
    elif end_hours % 8 < 4:
        add_end_hours = add_end_hours - 1

    end_datetime = end_date.strftime("%Y-%m-%d") + 'T' + str(add_end_hours) + ':59:59'

    return start_datetime, end_datetime


def create_new_asssignment(assignment_value, task_id, res_id, work_volume, start_date, end_date, units):
    assignment_element = ET.Element(tag_prefix + 'Assignment')

    uid_elem = ET.Element(tag_prefix + 'UID')
    uid_elem.text = str(assignment_value)
    assignment_element.append(uid_elem)

    elem = ET.Element(tag_prefix + 'GUID')
    elem.text = str(uuid4())
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'TaskID')
    elem.text = str(task_id)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'ResourceID')
    elem.text = str(res_id)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'PercentWorkComplete')
    elem.text = str(0)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'Finish')
    elem.text = str(end_date)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'HasFixedRateUnits')
    elem.text = str(1)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'FixedMaterial')
    elem.text = str(0)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'LevelingDelayFormat')
    elem.text = str(7)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'RemainingWork')
    elem.text = 'PT' + work_volume + 'H0M0S'
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'Start')
    elem.text = str(start_date)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'Units')
    elem.text = str(units)
    assignment_element.append(elem)

    elem = ET.Element(tag_prefix + 'Work')
    elem.text = 'PT' + work_volume + 'H0M0S'
    assignment_element.append(elem)

    elem_extended = ET.Element(tag_prefix + 'ExtendedAttribute')

    elem = ET.Element(tag_prefix + 'FieldID')
    elem.text = str(255852770)
    elem_extended.append(elem)

    elem = ET.Element(tag_prefix + 'Value')
    elem.text = "{}"
    elem_extended.append(elem)

    assignment_element.append(elem_extended)
    # assignment_element.append(ET.Element(tag_prefix + '/ExtendedAttribute'))

    return assignment_element


def get_works_info(filepath: str):
    # Parse XML file with the project's data
    parser = ET.XMLParser(encoding="utf-8")
    project_tree = ET.parse(filepath, parser=parser)
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

    project_tasks = project_root.find(tag_prefix + 'Tasks'). \
        findall(tag_prefix + 'Task')

    # Tasks info preparation
    for task_info in project_tasks:

        task_id = task_info.find(tag_prefix + 'UID').text
        input_data['activity_id'].append(task_id)

        task_name = task_info.find(tag_prefix + 'Name').text
        input_data['activity_name'].append(task_name)

        granular_name = task_name.split(' ')[0]
        if granular_name not in works_with_spec:
            if granular_name == 'Веха':
                input_data['granular_name'].append('Веха')
            else:
                input_data['granular_name'].append('Общая')
        else:
            input_data['granular_name'].append(granular_name)

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

    # Restructuring DataFrame and create service vertices for the beginning and ending of each block
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
    # for i in project_df.index:
    #     if project_df.loc[i, 'granular_name'] == 'Веха':
    #         project_df.loc[i, 'is_service'] = 1

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

    df_service = project_df[(project_df['is_executable'] == 0) & (project_df['is_service'] == 0)]

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

    return df_for_scheduling, df_service, project_wbs_levels, successors_dict


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


def get_project_calendar(path_to_input_xml: str):
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
                holidays_lst.extend(list(generate_dates(start_date, end_date)))
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
                    holidays.add(x)
            else:
                for x in list(generate_dates(start_date, end_date)):
                    if x in holidays:
                        holidays.remove(x)

    return working_days, holidays


def convert_dates_in_schedule(project_df, project_start_date):
    start_dates = []
    end_dates = []

    for i in project_df.index:
        start_time = int(project_df.loc[i, 'start'])
        end_time = int(project_df.loc[i, 'finish'])
        datetimes = get_datetime_string(project_start_date, start_time, end_time)
        start_dates.append(datetimes[0])
        end_dates.append(datetimes[1])

    project_df['start_date'] = start_dates
    project_df['finish_date'] = end_dates

    return project_df


def process_schedule(schedule_df, structure_info):

    structure_df, project_wbs_levels, successors_dict = structure_info
    schedule_df = schedule_df[['task_id', 'task_name', 'volume', 'measurement',
                               'cost', 'start', 'finish',
                               'start_date', 'finish_date', 'duration', 'workers']]
    schedule_df = schedule_df.rename(columns={'task_id': 'activity_id',
                                              'task_name': 'activity_name'})
    schedule_df['is_executable'] = [1] * len(schedule_df)
    schedule_df['activity_id'] = [str(x) for x in schedule_df['activity_id']]

    structure_df = structure_df[['activity_id', 'activity_name', 'volume', 'measurement']]
    structure_df['cost'] = [0] * len(structure_df)
    structure_df['start'] = [0] * len(structure_df)
    structure_df['finish'] = [0] * len(structure_df)
    structure_df['start_date'] = [''] * len(structure_df)
    structure_df['finish_date'] = [''] * len(structure_df)
    structure_df['duration'] = [-1] * len(structure_df)
    structure_df['workers'] = ['{}'] * len(structure_df)
    structure_df['is_executable'] = [0] * len(structure_df)

    project_df2 = pd.concat([schedule_df, structure_df])

    project_df2['activity_id'] = [str(x) for x in project_df2['activity_id']]
    project_df2.to_csv('project_df_after_merging.csv', index=False)

    project_df2 = project_df2.set_index('activity_id')

    for wbs_lvl in range(max(project_wbs_levels.keys()), 0, -1):
        for task_id in project_wbs_levels[wbs_lvl]:
            if project_df2.loc[task_id, 'is_executable'] == 0:
                # print(task_id)
                min_id = -1
                min_val = 2000000
                max_id = -1
                max_val = -1
                cost = 0
                for inner_task_id in successors_dict[task_id]:
                    cost += project_df2.loc[inner_task_id, 'cost']
                    if project_df2.loc[inner_task_id, 'start'] < min_val:
                        min_val = project_df2.loc[inner_task_id, 'start']
                        min_id = inner_task_id
                    if project_df2.loc[inner_task_id, 'finish'] > max_val:
                        max_val = project_df2.loc[inner_task_id, 'finish']
                        max_id = inner_task_id

                project_df2.loc[task_id, 'start'] = project_df2.loc[min_id, 'start']
                project_df2.loc[task_id, 'start_date'] = project_df2.loc[min_id, 'start_date']
                project_df2.loc[task_id, 'finish'] = project_df2.loc[min_id, 'finish']
                project_df2.loc[task_id, 'finish_date'] = project_df2.loc[max_id, 'finish_date']
                project_df2.loc[task_id, 'duration'] = int(max_val - min_val)
                project_df2.loc[task_id, 'cost'] = cost

                # if min_id != -1:
                #     project_df2.loc[task_id, 'start'] = project_df2.loc[min_id, 'start']
                #     project_df2.loc[task_id, 'start_date'] = project_df2.loc[min_id, 'start_date']
                # else:
                #     project_df2.loc[task_id, 'start'] = -1
                #     project_df2.loc[task_id, 'start_date'] = ''
                # if max_id != -1:
                #     project_df2.loc[task_id, 'finish'] = project_df2.loc[min_id, 'finish']
                #     project_df2.loc[task_id, 'finish_date'] = project_df2.loc[max_id, 'finish_date']
                #     if min_id != 1:
                #         project_df2.loc[task_id, 'duration'] = int(max_val - min_val)
                #
                # project_df2.loc[task_id, 'cost'] = cost

    # project_df2 = project_df2[project_df2['finish'] != -1]
    # project_df2 = project_df2[project_df2['start'] != -1]

    project_df2 = project_df2.reset_index()
    return project_df2, list(project_wbs_levels[1])[0]


def schedule_csv_to_xml(schedule_df, project_id, filepath, output_xml_name):
    # Parse XML file with the project's data
    parser = ET.XMLParser(encoding="utf-8")
    project_tree = ET.parse(filepath, parser=parser)
    project_root = project_tree.getroot()

    schedule_df = schedule_df.set_index('activity_id')
    schedule_df['workers'] = [literal_eval(x) for x in schedule_df['workers']]

    # Fix project info
    project_root.find(tag_prefix + 'StartDate').text = schedule_df.loc[project_id, 'start_date']
    project_root.find(tag_prefix + 'FinishDate').text = schedule_df.loc[project_id, 'finish_date']
    project_root.find(tag_prefix + 'CurrentDate').text = current_date = '2024-03-14T08:38:12'

    # Fix tasks info
    for task_info in project_root.find(tag_prefix + 'Tasks'). \
            findall(tag_prefix + 'Task'):
        task_id = task_info.find(tag_prefix + 'UID').text
        task_info.find(tag_prefix + 'Duration').text = 'PT' + str(schedule_df.loc[task_id, 'duration']) + 'H0M0S'
        if not task_info.find(tag_prefix + 'RemainingDuration') is None:
            task_info.find(tag_prefix + 'RemainingDuration').text = 'PT' + str(
                schedule_df.loc[task_id, 'duration']) + 'H0M0S'
        task_info.find(tag_prefix + 'Start').text = schedule_df.loc[task_id, 'start_date']
        task_info.find(tag_prefix + 'Finish').text = schedule_df.loc[task_id, 'finish_date']

    assignments_lst = project_root.find(tag_prefix + 'Assignments'). \
        findall(tag_prefix + 'Assignment')
    assignments_element = project_root.find(tag_prefix + 'Assignments')
    n_assignments = len(assignments_lst)

    workers_lst = get_resources_info(filepath)

    get_resource_id_by_name = {}
    for worker in workers_lst:
        get_resource_id_by_name[worker['name']] = worker['id']

    assignment_id = 1
    for task_id in schedule_df.index:
        if 'Веха' not in schedule_df.loc[task_id, 'activity_name']:
            for res_name in schedule_df.loc[task_id, 'workers']:
                if assignment_id <= n_assignments:
                    assignment = assignments_lst[assignment_id - 1]
                    assignment.find(tag_prefix + 'GUID').text = str(uuid4())
                    assignment.find(tag_prefix + 'TaskUID').text = str(task_id)
                    assignment.find(tag_prefix + 'ResourceUID').text = str(get_resource_id_by_name[res_name])
                    assignment.find(tag_prefix + 'Start').text = schedule_df.loc[task_id, 'start_date']
                    assignment.find(tag_prefix + 'Finish').text = schedule_df.loc[task_id, 'finish_date']
                    assignment.find(tag_prefix + 'Units').text = str(schedule_df.loc[task_id, 'workers'][res_name])
                else:
                    assignments_element.append(create_new_asssignment(str(assignment_id),
                                                                      str(task_id),
                                                                      str(get_resource_id_by_name[res_name]),
                                                                      str(int(schedule_df.loc[task_id, 'volume'])),
                                                                      schedule_df.loc[task_id, 'start_date'],
                                                                      schedule_df.loc[task_id, 'finish_date'],
                                                                      str(schedule_df.loc[task_id, 'workers'][res_name])))
                assignment_id += 1

    project_tree.write(output_xml_name, encoding="utf-8")