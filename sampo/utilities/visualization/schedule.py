from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px

from sampo.utilities.visualization.base import VisualizationMode, visualize


def schedule_gant_chart_fig(schedule_dataframe: pd.DataFrame,
                            visualization: VisualizationMode,
                            remove_service_tasks: bool = False,
                            fig_file_name: Optional[str] = None):
    """
    Creates and saves a gant chart of the scheduled tasks to the specified path.

    :param fig_file_name:
    :param visualization:
    :param remove_service_tasks:
    :param schedule_dataframe: Pandas DataFrame with the information about schedule
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

    def create_zone_row(i, zone_names, zone) -> dict:
        return {'idx': i,
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

    schedule_dataframe = pd.concat([schedule_dataframe, pd.DataFrame.from_records(access_cards)])

    schedule_dataframe['color'] = schedule_dataframe[['task_name', 'contractor']] \
        .apply(lambda r: 'Defect' if ':' in r['task_name'] else r['contractor'], axis=1)
    schedule_dataframe['idx'] = (schedule_dataframe[['idx', 'task_name']]
                                 .apply(lambda r: schedule_dataframe[schedule_dataframe['task_name'] ==
                                                                     r['task_name'].split(':')[0]]['idx'].iloc[0]
                                 if ':' in r['task_name'] else r['idx'], axis=1))

    # add one time unit to the end should remove hole within the immediately close tasks
    schedule_dataframe['vis_finish'] = schedule_dataframe[['start', 'finish', 'duration']] \
        .apply(lambda r: r['finish'] + timedelta(1) if r['duration'] > 0 else r['finish'], axis=1)
    schedule_dataframe['vis_start'] = schedule_dataframe['start']
    schedule_dataframe['finish'] = schedule_dataframe['finish'].apply(lambda x: x.strftime('%e %b %Y'))
    schedule_dataframe['start'] = schedule_dataframe['start'].apply(lambda x: x.strftime('%e %b %Y'))

    fig = px.timeline(schedule_dataframe, x_start='vis_start', x_end='vis_finish', y='idx', hover_name='task_name',
                      color=schedule_dataframe.loc[:, 'color'],
                      hover_data={'vis_start': False,
                                  'vis_finish': False,
                                  'start': True,
                                  'finish': True,
                                  'task_name_mapped': True,
                                  'cost': True,
                                  'volume': True,
                                  'measurement': True,
                                  'workers': True,
                                  'zone_information': True},
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
    fig.update_layout(height=1000)

    return visualize(fig, mode=visualization, file_name=fig_file_name)
