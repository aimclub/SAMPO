from datetime import timedelta
from operator import attrgetter
from typing import Optional

import pandas as pd
import plotly.express as px
from matplotlib.figure import Figure

from sampo.schemas import Time, WorkGraph, WorkTimeEstimator, GraphNode
from sampo.utilities.visualization.base import VisualizationMode, visualize


def schedule_gant_chart_fig(schedule_dataframe: pd.DataFrame,
                            visualization: VisualizationMode,
                            critical_path: list[GraphNode] = None,
                            remove_service_tasks: bool = False,
                            fig_file_name: Optional[str] = None,
                            color_type: str = 'contractor') -> Figure | None:
    """
    Creates and saves a gant chart of the scheduled tasks to the specified path.

    :param fig_file_name:
    :param visualization:
    :param wg:
    :param remove_service_tasks:
    :param schedule_dataframe: Pandas DataFrame with the information about schedule
    :param color_type defines what tasks color means
    """
    if remove_service_tasks:
        schedule_dataframe = schedule_dataframe.loc[~schedule_dataframe.loc[:, 'task_name'].str.contains('марке')]
        schedule_dataframe = schedule_dataframe.loc[schedule_dataframe.loc[:, 'volume'] >= 0.1]

    schedule_dataframe = schedule_dataframe.rename({'workers': 'workers_dict'}, axis=1)
    schedule_dataframe.loc[:, 'workers'] = schedule_dataframe.loc[:, 'workers_dict']\
        .apply(lambda x: x.replace(", '", ", <br>'"))

    schedule_start = schedule_dataframe.loc[:, 'start'].min()
    schedule_finish = schedule_dataframe.loc[:, 'finish'].max()
    visualization_start_delta = timedelta(days=2)
    visualization_finish_delta = timedelta(days=(schedule_finish - schedule_start).days // 3)

    def create_delivery_row(i, mat_name, material) -> dict:
        return {'idx': i,
                'task_id': 'None',
                'contractor': material[-1],
                'cost': 0,
                'volume': material[0],
                'duration': 0,
                'measurement': 'unit',
                'workers_dict': '',
                'workers': '',
                'task_name_mapped': mat_name,
                'task_name': '',
                'zone_information': '',
                'start': timedelta(material[1].value) + schedule_start,
                'finish': timedelta(material[2].value) + schedule_start}

    sworks = schedule_dataframe['scheduled_work_object'].copy()
    idx = schedule_dataframe['idx'].copy()

    def get_delivery_info(swork) -> str:
        return '<br>' + '<br>'.join([f'{mat[0][0]}: {mat[0][1]}' for mat in swork.materials.delivery.values()])

    schedule_dataframe['material_information'] = sworks.apply(get_delivery_info)

    mat_delivery_row = []

    # create material delivery information
    for i, swork in zip(idx, sworks):
        delivery = swork.materials.delivery
        for name, mat_info in delivery.items():
            if delivery:
                mat_delivery_row.append(create_delivery_row(i, name, mat_info[0]))

    def create_zone_row(i, zone_names, zone) -> dict:
        return {'idx': i,
                'task_id': 'None',
                'contractor': 'Access cards',
                'cost': 0,
                'volume': 0,
                'duration': 0,
                'measurement': 'unit',
                'workers_dict': '',
                'workers': '',
                'task_name_mapped': zone_names,
                'task_name': '',
                'zone_information': '',
                'start': timedelta(int(zone.start_time)) + schedule_start,
                'finish': timedelta(int(zone.end_time)) + schedule_start}

    sworks = schedule_dataframe['scheduled_work_object'].copy()
    idx = schedule_dataframe['idx'].copy()

    def get_zone_usage_info(swork) -> str:
        return '<br>' + '<br>'.join([f'{zone.name}: {zone.to_status}' for zone in swork.zones_pre])

    schedule_dataframe['zone_information'] = sworks.apply(get_zone_usage_info)

    access_cards = []

    # create zone information
    for i, swork in zip(idx, sworks):
        zone_names = '<br>' + '<br>'.join([zone.name for zone in swork.zones_pre])
        for zone in swork.zones_pre:
            access_cards.append(create_zone_row(i, zone_names, zone))
        zone_names = '<br>' + '<br>'.join([zone.name for zone in swork.zones_post])
        for zone in swork.zones_post:
            access_cards.append(create_zone_row(i, zone_names, zone))

    schedule_dataframe = pd.concat([schedule_dataframe, pd.DataFrame.from_records(access_cards), pd.DataFrame.from_records(mat_delivery_row)])

    if color_type == 'contractor':
        schedule_dataframe['color'] = schedule_dataframe[['task_name', 'contractor']] \
            .apply(lambda r: 'Defect' if ':' in r['task_name'] else r['contractor'], axis=1)
    elif color_type == 'priority':
        schedule_dataframe['color'] = schedule_dataframe['scheduled_work_object'].apply(lambda x: f'Priority {x.priority}')
    elif color_type == 'critical_path':
        critical_path_nodes = set(map(attrgetter('id'), critical_path))
        schedule_dataframe['color'] = schedule_dataframe['scheduled_work_object'] \
                .apply(lambda x: 'Critical path' if x.id in critical_path_nodes else 'Not critical path')

    schedule_dataframe['idx'] = (schedule_dataframe[['idx', 'task_name']]
                                 .apply(lambda r: schedule_dataframe[schedule_dataframe['task_name'] ==
                                                                     r['task_name'].split('&')[0]]['idx'].iloc[0]
                                 if ':' in r['task_name'] else r['idx'], axis=1))


    fig = px.timeline(schedule_dataframe, x_start='start', x_end='finish', y='idx', hover_name='task_name',
                      color=schedule_dataframe.loc[:, 'color'] if 'color' in schedule_dataframe.columns else None,
                      hover_data={'task_id': True,
                                  'start': True,
                                  'finish': True,
                                  'task_name_mapped': True,
                                  'cost': True,
                                  'volume': True,
                                  'measurement': True,
                                  'workers': True,
                                  'zone_information': True,
                                  'material_information': True},
                      title=f"{'Project tasks - Gant chart'}",
                      category_orders={'idx': list(schedule_dataframe.idx)},
                      text='task_name')

    fig.update_traces(textposition='outside')

    fig.update_yaxes(showticklabels=False, title_text='Project tasks',
                     type='category')

    fig.update_xaxes(type='date',
                     range=[schedule_start - visualization_start_delta,
                            schedule_finish + visualization_finish_delta],
                     title_text='Date')

    fig.update_layout(autosize=True, font_size=12)
    # fig.update_layout(height=1000)

    return visualize(fig, mode=visualization, file_name=fig_file_name)
