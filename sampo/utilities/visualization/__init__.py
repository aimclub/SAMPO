from matplotlib.figure import Figure

from sampo.schemas import Schedule
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.resources import resource_employment_fig, EmploymentFigType
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.utilities.visualization.work_graph import work_graph_fig


class ScheduleVisualization:
    def __init__(self, schedule: Schedule, start_date: str):
        self._schedule = schedule.merged_stages_datetime_df(start_date)

    def gant_chart(self, visualization_mode: VisualizationMode = VisualizationMode.ReturnFig) -> Figure | None:
        return schedule_gant_chart_fig(self._schedule, visualization=visualization_mode)

    def date_labeled_resource_chart(self, visualization_mode: VisualizationMode = VisualizationMode.ReturnFig) \
            -> Figure | None:
        return resource_employment_fig(self._schedule,
                                       fig_type=EmploymentFigType.DateLabeled,
                                       vis_mode=visualization_mode)

    def work_labeled_resource_chart(self, visualization_mode: VisualizationMode = VisualizationMode.ReturnFig) \
            -> Figure | None:
        return resource_employment_fig(self._schedule,
                                       fig_type=EmploymentFigType.WorkLabeled,
                                       vis_mode=visualization_mode)
