from datetime import datetime
import warnings

from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler import GeneticScheduler, HEFTScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme

from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

from work_time_estimator import WorkEstimator, CalendarBasedWorkEstimator

from xml_parser import get_works_info, get_contractors_info, get_project_calendar, \
    convert_dates_in_schedule, process_schedule, schedule_csv_to_xml

from experiments.hackathon.skills_resource_optimizer import GreedyMinimalMultiSkillResourceOptimizer

warnings.filterwarnings("ignore")

filepath = './sber_task.xml'
output_path = 'scheduled_xml.xml'
project_start_date = datetime(2024, 2, 5)

df, *structure_info = get_works_info(filepath)
contractors = get_contractors_info(filepath)
project_business_calendar = get_project_calendar(filepath)

scheduling_pipeline = SchedulingPipeline.create()
scheduling_pipeline = scheduling_pipeline.wg(wg=df,
                                             all_connections=True,
                                             change_connections_info=False)

scheduling_pipeline = scheduling_pipeline.contractors(contractors).lag_optimize(LagOptimizationStrategy.TRUE)

# project_work_estimator = WorkEstimator()

project_work_estimator = CalendarBasedWorkEstimator(project_calendar=project_business_calendar,
                                                    working_hours_cnt=8,
                                                    start_date=project_start_date)

scheduling_pipeline = scheduling_pipeline.work_estimator(project_work_estimator)
genetic_scheduler_with_estimator = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                                    mutate_order=0.05,
                                                    mutate_resources=0.005,
                                                    sgs_type=ScheduleGenerationScheme.Parallel,
                                                    only_lft_initialization=True,
                                                    work_estimator=project_work_estimator
                                                    )
greedy_resource_optimizer = GreedyMinimalMultiSkillResourceOptimizer()
heft_scheduler = HEFTScheduler(resource_optimizer=greedy_resource_optimizer,
                               work_estimator=project_work_estimator)
# scheduling_project = scheduling_pipeline.schedule(genetic_scheduler_with_estimator).finish()[0]
scheduling_project = scheduling_pipeline.schedule(heft_scheduler).finish()[0]

raw_project_schedule = scheduling_project.schedule
pure_schedule = raw_project_schedule.pure_schedule_df
schedule_df = convert_dates_in_schedule(pure_schedule, project_start_date)
schedule_df.to_csv('scheduled.csv', index=0)

schedule_csv_to_xml(*process_schedule(schedule_df, structure_info), filepath, output_path)


# FOR OUR DEBUG
print(raw_project_schedule.execution_time.value)

project_schedule = raw_project_schedule.merged_stages_datetime_df('2023-02-05')

schedule_fig = schedule_gant_chart_fig(schedule_dataframe=project_schedule,
                                       visualization=VisualizationMode.ShowFig,
                                       remove_service_tasks=False)
