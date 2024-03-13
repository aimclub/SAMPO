import xml.etree.ElementTree as ET

from datetime import timedelta, date
from business_calendar import Calendar, MO, TU, WE, TH, FR, SA, SU

tag_prefix = '{http://schemas.microsoft.com/project}'

week_days_by_id = {
    '1': SU,
    '2': MO,
    '3': TU,
    '4': WE,
    '5': TH,
    '6': FR,
    '7': SA}


def get_calendar_info_from_xml(path_to_input_xml):
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


def get_project_calendar(path_to_input_xml: str) -> Calendar:
    main_calendar = get_calendar_info_from_xml(path_to_input_xml)

    # Get information about working days and holidays from WeekDays tag
    week_days = main_calendar.find(tag_prefix + 'WeekDays').findall(tag_prefix + 'WeekDay')

    working_days = []
    holidays_lst = []

    for week_day in week_days:
        if week_day.find(tag_prefix + 'DayWorking').text == '1':
            working_days.append(week_days_by_id[week_day.find(tag_prefix + 'DayType').text])
        elif week_day.find(tag_prefix + 'DayType').text == '0':
            time_period = week_day.find(tag_prefix + 'TimePeriod')
            if not time_period is None:
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
