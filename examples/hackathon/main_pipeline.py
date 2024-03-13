import pandas as pd
from ast import literal_eval

from sampo.pipeline import SchedulingPipeline
from sampo.schemas.contractor import Contractor
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler import GeneticScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme

from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.utilities.visualization.resources import resource_employment_fig, EmploymentFigType

from work_time_estimator import WorkEstimator


df = pd.read_csv('./' + 'works_info.csv')

df['req_volume'] = [literal_eval(x) for x in df['req_volume']]
df['min_req'] = [literal_eval(x) for x in df['min_req']]
df['max_req'] = [literal_eval(x) for x in df['max_req']]
scheduling_pipeline = SchedulingPipeline.create()
scheduling_pipeline = scheduling_pipeline.wg(wg=df,
                                             all_connections=True,
                                             change_connections_info=False)

contractors = [Contractor.load('./', 'project_contractor')]
scheduling_pipeline = scheduling_pipeline.contractors(contractors)
scheduling_pipeline = scheduling_pipeline.lag_optimize(LagOptimizationStrategy.TRUE)

project_work_estimator = WorkEstimator()
scheduling_pipeline = scheduling_pipeline.work_estimator(project_work_estimator)
genetic_scheduler_with_estimator = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                                    mutate_order=0.05,
                                                    mutate_resources=0.005,
                                                    sgs_type=ScheduleGenerationScheme.Serial,
                                                    work_estimator=project_work_estimator
                                                    )
scheduling_project = scheduling_pipeline.schedule(genetic_scheduler_with_estimator).finish()[0]

raw_project_schedule = scheduling_project.schedule

project_schedule = raw_project_schedule.merged_stages_datetime_df('2022-09-01')

schedule_fig = schedule_gant_chart_fig(schedule_dataframe=project_schedule,
                                       visualization=VisualizationMode.ShowFig,
                                       remove_service_tasks=False)
