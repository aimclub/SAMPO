from ast import literal_eval
from collections import defaultdict
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import chain
from operator import itemgetter, attrgetter
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import plotly.graph_objects
from matplotlib import pyplot as plt
from pandas import DataFrame, Timestamp
from plotly import express as px

from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ResourceSchedule, ScheduleWorkDict, Schedule
from sampo.schemas.time import Time
from sampo.utilities.visualization.base import VisualizationMode, visualize
from sampo.utilities.visualization.work_graph import SIZE_LIMIT, DPI_LIMIT, DEFAULT_DPI

SPLITTER = '#'


class EmploymentFigType(Enum):
    Classic = auto()
    Grouped = auto()
    DateLabeled = auto()
    WorkLabeled = auto()


def resource_employment_fig(schedule: Union[DataFrame, Schedule],
                            fig_type: EmploymentFigType = EmploymentFigType.Classic,
                            vis_mode: VisualizationMode = VisualizationMode.ShowFig,
                            file_name: Optional[str] = None,
                            project_start: datetime = datetime(year=2020, month=1, day=1)) \
        -> Optional[Union[plt.Figure, plotly.graph_objects.Figure]]:

    graph_data = (convert_schedule_df(schedule, fig_type)
                  if isinstance(schedule, DataFrame)
                  else get_schedule_df(schedule.to_schedule_work_dict, fig_type, project_start)) \
        if fig_type in [EmploymentFigType.WorkLabeled, EmploymentFigType.DateLabeled] \
        else get_workers_intervals(schedule, fig_type is EmploymentFigType.Grouped)

    return create_employment_fig(graph_data, fig_type, vis_mode, file_name)


def create_employment_fig(resources: Union[DataFrame, ResourceSchedule],
                          fig_type: EmploymentFigType,
                          vis_mode: VisualizationMode,
                          file_name: Optional[str] = '') \
        -> Optional[Union[plt.Figure, plotly.graph_objects.Figure]]:
    """

    :param resources:
    :param fig_type:
    :param vis_mode:
    :param file_name:
    :return:
    """

    if fig_type in [EmploymentFigType.WorkLabeled, EmploymentFigType.DateLabeled]:
        assert isinstance(resources, DataFrame), \
            f'Wrong data format. Expected DataFrame, got {type(resources)} instead.'

        if fig_type == EmploymentFigType.WorkLabeled:
            fig = px.timeline(data_frame=resources,
                              title=f"{'Resource load by tasks - chart'}",
                              x_start='time_start', x_end='time_end',
                              y='resource',
                              color='count',
                              hover_data={'count': True, 'time_start': True, 'time_end': False, 'resource': True},
                              text='count',
                              color_continuous_scale=px.colors.sequential.Agsunset_r)
            fig.update_traces(textposition='inside')
            works = {}
            for _, row in resources.iterrows():
                if row['time_start'] not in works:
                    works[row['time_start']] = row['work']

            w_lst = sorted(works.items(), key=itemgetter(0))

            fig.update_xaxes(tickangle=75,
                             # category_orders={'time_start': [x[1] for x in w_lst]},
                             tickmode='array',
                             tickvals=[x[0] for x in w_lst],
                             ticktext=[x[1] for x in w_lst])
        else:
            fig = px.timeline(data_frame=resources,
                              x_start='time_start', x_end='time_end',
                              title=f"{'Resource load by days - chart'}",
                              y='resource',
                              color='count',
                              hover_data=['count'],
                              text='count',
                              color_continuous_scale=px.colors.sequential.Agsunset_r)
            fig.update_traces(textposition='inside')
        return visualize(fig, vis_mode, file_name)

    assert isinstance(resources, ResourceSchedule), \
        f'Wrong data format. Expected ResourceSchedule, got {type(resources)} instead.'

    # 1. tune plot parameters according to the data size
    proposed_height = max(10, int(len(resources) * 0.2))  # used, if size limit does not exceed
    size = (20, proposed_height if proposed_height <= SIZE_LIMIT else SIZE_LIMIT)

    increased_dpi = DEFAULT_DPI + abs((proposed_height - SIZE_LIMIT) / 200)  # used to increase readability in big figs
    dpi = DEFAULT_DPI if proposed_height <= SIZE_LIMIT else min((increased_dpi, DPI_LIMIT))

    # 2. plot the data
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    for name, worker_schedule in resources.items():
        for worker_time in worker_schedule:
            ax.barh(name, left=worker_time[0], width=worker_time[1] - worker_time[0])

    for i in range(len(resources) + 1):
        ax.axhline(i - 0.5, color='black')

    plt.tight_layout()
    return visualize(fig, vis_mode, file_name)


def get_resources(item):
    workers: Union[str, Dict[str, int]] = item['workers']
    resources: Dict[str, int] = item['workers_dict'] \
        if 'workers_dict' in item.index \
        else (literal_eval(workers) if isinstance(workers, str) else workers)
    return resources


def convert_schedule_df(schedule: DataFrame, fig_type: EmploymentFigType) -> DataFrame:
    first_day = schedule['start'].min()
    first_day = datetime(year=first_day.year, month=first_day.month, day=first_day.day)

    if fig_type is EmploymentFigType.DateLabeled:
        total_days = (schedule['finish'].max() - first_day).days + 1
        resource_schedule: Dict[str, np.ndarray] = {}
        for _, item in schedule.iterrows():
            start: Timestamp = (item['start'] - first_day).days
            finish: Timestamp = (item['finish'] - first_day).days
            resources: Dict[str, int] = get_resources(item)
            for name in resources:
                if name not in resource_schedule:
                    resource_schedule[name] = np.array([0] * total_days)
                resource_schedule[name][start:finish + 1] += resources[name]

        data = list(chain(*[[(timedelta(days=i), timedelta(days=i + 1), r, c) for i, c in enumerate(s) if c > 0]
                            for r, s in resource_schedule.items()]))
        result = DataFrame.from_records(data, columns=['time_start', 'time_end', 'resource', 'count'])
        result.loc[:, ['time_start', 'time_end']] += first_day
        return result

    if fig_type is EmploymentFigType.WorkLabeled:
        resource_schedule: Dict[str, List[Tuple[int, str, int]]] = {}
        schedule.index = list(range(schedule.shape[0]))

        i = 0
        for _, item in schedule.iterrows():
            resources = get_resources(item)
            if len(resources) == 0:
                continue
            w_name = item['task_name']
            for name in resources:
                if resources[name] > 0:
                    if name not in resource_schedule:
                        resource_schedule[name] = []
                    resource_schedule[name].append((i, w_name, resources[name]))
            i += 1

        data = list(chain(*[[(timedelta(days=i), timedelta(days=i + 1), w, r, c) for i, w, c in s]
                            for r, s in resource_schedule.items()]))
        result = DataFrame.from_records(data, columns=['time_start', 'time_end', 'work', 'resource', 'count'])
        result.loc[:, ['time_start', 'time_end']] += first_day
        return result


def get_schedule_df(schedule: ScheduleWorkDict, fig_type: EmploymentFigType, project_start: datetime) -> DataFrame:
    if fig_type == EmploymentFigType.DateLabeled:
        max_time = max(schedule.values(), key=attrgetter('finish_time')).finish_time + 1
        resource_schedule: Dict[str, List[int]] = {}
        for item in schedule.values():
            start: Time = item.start_time
            finish: Time = item.finish_time
            resources: List[Worker] = item.workers
            for worker in resources:
                if worker.name not in resource_schedule:
                    resource_schedule[worker.name] = np.array([0] * int(max_time))
                resource_schedule[worker.name][start:finish + 1] += worker.count

        data = list(chain(*[[(timedelta(days=i), timedelta(days=i + 1), r, c) for i, c in enumerate(s) if c > 0]
                            for r, s in resource_schedule.items()]))
        result = DataFrame.from_records(data, columns=['time_start', 'time_end', 'resource', 'count'])
        result.loc[:, ['time_start', 'time_end']] += project_start
        return result

    if fig_type == EmploymentFigType.WorkLabeled:
        resource_schedule: Dict[str, List[Tuple[int, str, int]]] = {}
        for i, (work, item) in enumerate(sorted(list(schedule.items()), key=lambda x: x[1].start_time)):
            resources: List[Worker] = item.workers
            w_name = item.name
            for worker in resources:
                if worker.count > 0:
                    if worker.name not in resource_schedule:
                        resource_schedule[worker.name] = []
                    resource_schedule[worker.name].append((i, w_name, worker.count))

        data = list(chain(*[[(timedelta(days=i), timedelta(days=i + 1), w, r, c) for i, w, c in s]
                            for r, s in resource_schedule.items()]))
        result = DataFrame.from_records(data, columns=['time_start', 'time_end', 'work', 'resource', 'count'])
        result.loc[:, ['time_start', 'time_end']] += project_start
        return result


def get_workers_intervals(schedule: Schedule, group_workers_by_specializations: bool = False) \
        -> ResourceSchedule:
    resources_time_intervals_dict: Dict[Time, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in schedule.works:
        start: Time = item.start_time
        finish: Time = item.finish_time
        workers: List[Worker] = item.workers
        for worker in workers:
            resources_time_intervals_dict[start][worker.name] -= worker.count
            resources_time_intervals_dict[finish][worker.name] += worker.count

    min_resources: Dict[str, int] = defaultdict(int)
    for milestone, resources in resources_time_intervals_dict.items():
        resources_time_intervals_dict[milestone] = dict(resources)
        for name in resources:
            min_resources[name] = min(min_resources[name], resources[name])
    resources_time_intervals_dict = dict(resources_time_intervals_dict)

    resources_time_intervals: List[Time, Dict[str, int]] = list(resources_time_intervals_dict.items())
    resources_time_intervals = sorted(resources_time_intervals, key=itemgetter(0))
    workers_intervals = defaultdict(list)
    for index, (start, resources) in enumerate(resources_time_intervals[:-1]):
        finish = resources_time_intervals[index + 1]
        for name in resources:
            resources[name] -= min_resources[name]
            if group_workers_by_specializations:
                workers_intervals[name].append((start, finish))
            else:
                for w_index in range(resources[name]):
                    workers_intervals[f'{name}{SPLITTER}{w_index}'].append((start, finish))

    if group_workers_by_specializations:
        workers_intervals = dict(sorted(dict(workers_intervals).items(),
                                        key=lambda item: item[0]))
    else:
        workers_intervals = dict(sorted(dict(workers_intervals).items(),
                                        key=lambda item: (item[0].split(SPLITTER)[0], int(item[0].split(SPLITTER)[1]))))
    return workers_intervals
