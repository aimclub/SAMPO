from matplotlib.figure import Figure

from sampo.pipeline import PipelineError
from sampo.schemas import Schedule, WorkGraph, ScheduledProject
from sampo.utilities.visualization.base import VisualizationMode, visualize
from sampo.utilities.visualization.resources import resource_employment_fig, EmploymentFigType
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.utilities.visualization.work_graph import work_graph_fig


class ScheduleVisualization:
    def __init__(self, schedule: Schedule, start_date: str):
        self._schedule = schedule.merged_stages_datetime_df(start_date)
        self._shape = 10, 10

    def fig(self, shape: tuple[int, int]) -> 'ScheduleVisualization':
        self._shape = shape
        return self

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


class WorkGraphVisualization:
    def __init__(self, wg: WorkGraph):
        self._wg = wg
        self._shape = 10, 10

    def fig(self, shape: tuple[int, int]) -> 'WorkGraphVisualization':
        self._shape = shape
        return self

    def work_graph_chart(self, visualization_mode: VisualizationMode = VisualizationMode.ReturnFig) -> Figure | None:
        return visualize(work_graph_fig(self._wg, self._shape), visualization_mode)


class Visualization:

    def __init__(self):
        self.wg_vis = None
        self.schedule_vis = None

    @staticmethod
    def from_project(project: ScheduledProject, start_date: str) -> 'Visualization':
        vis = Visualization()
        vis.wg(project.wg)
        vis.schedule(project.schedule, start_date)
        return vis

    def wg(self, wg: WorkGraph):
        self.wg_vis = WorkGraphVisualization(wg)

    def schedule(self, schedule: Schedule, start_date: str):
        self.schedule_vis = ScheduleVisualization(schedule, start_date)

    def shape(self, fig_shape: tuple[int, int]) -> 'Visualization':
        self.wg_vis.fig(fig_shape)
        self.schedule_vis.fig(fig_shape)
        return self

    def show_gant_chart(self) -> 'Visualization':
        if self.schedule_vis is None:
            raise PipelineError('Schedule not specified')
        self.schedule_vis.gant_chart(VisualizationMode.ShowFig)
        return self

    def show_resource_charts(self) -> 'Visualization':
        if self.schedule_vis is None:
            raise PipelineError('Schedule not specified')
        self.schedule_vis.date_labeled_resource_chart(VisualizationMode.ShowFig)
        self.schedule_vis.work_labeled_resource_chart(VisualizationMode.ShowFig)
        return self

    def show_work_graph(self) -> 'Visualization':
        if self.wg_vis is None:
            raise PipelineError('WorkGraph not specified')
        self.wg_vis.work_graph_chart(VisualizationMode.ShowFig)
        return self

    def get_all_figs(self):
        return {
            'date_labeled_resource_chart': self.schedule_vis.date_labeled_resource_chart(VisualizationMode.ReturnFig),
            'work_labeled_resource_chart': self.schedule_vis.work_labeled_resource_chart(VisualizationMode.ReturnFig),
            'gant_chart': self.schedule_vis.gant_chart(VisualizationMode.ReturnFig),
            'wg': self.wg_vis.work_graph_chart(VisualizationMode.ReturnFig)
        }
