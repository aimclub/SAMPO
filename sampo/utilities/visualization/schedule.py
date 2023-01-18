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

    fig = px.timeline(schedule_dataframe, x_start='start', x_end='finish', y='idx', hover_name='task_name',
                      color=schedule_dataframe.loc[:, 'contractor'],
                      hover_data=['cost', 'volume', 'measurement', 'workers'],
                      title=f"{'Project tasks - Gant chart'}",
                      category_orders={'idx': list(schedule_dataframe.idx)},
                      text='task_name')

    fig.update_traces(textposition='outside')

    fig.update_yaxes(showticklabels=False, title_text='Project tasks',
                     type="category")

    fig.update_xaxes(type="date",
                     range=[schedule_start - visualization_start_delta,
                            schedule_finish + visualization_finish_delta],
                     title_text='Date')

    fig.update_layout(autosize=True, font_size=12)

    return visualize(fig, mode=visualization, file_name=fig_file_name)
