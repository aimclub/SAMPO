from datetime import datetime
import warnings

from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler import GeneticScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme, TimeFitness

from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

from work_time_estimator import WorkEstimator, CalendarBasedWorkEstimator

from xml_parser import get_works_info, get_contractors_info, get_project_calendar

from fitness import MultiFitness, WeightedFitness, count_resources

warnings.filterwarnings("ignore")

filepath = './sber_task.xml'

df = get_works_info(filepath)
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
                                                    start_date=datetime(2024, 2, 5))

scheduling_pipeline = scheduling_pipeline.work_estimator(project_work_estimator)

fitness = TimeFitness()
# fitness = WeightedFitness(time_weight=0.5, cost_weight=0.5, resources_weight=0)
# fitness = WeightedFitness(time_weight=0.5, resources_weight=0.5, cost_weight=0)
# fitness = WeightedFitness(time_weight=0.33, resources_weight=0.33, cost_weight=0.33)
genetic_scheduler_with_estimator = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                                    mutate_order=0.05,
                                                    mutate_resources=0.005,
                                                    sgs_type=ScheduleGenerationScheme.Parallel,
                                                    only_lft_initialization=True,
                                                    work_estimator=project_work_estimator,
                                                    fitness_constructor=fitness,
                                                    )

scheduling_project = scheduling_pipeline.schedule(genetic_scheduler_with_estimator).finish()[0]

raw_project_schedule = scheduling_project.schedule

raw_project_schedule.pure_schedule_df.to_csv('scheduled.csv', index=0)

print(raw_project_schedule.execution_time.value)

# multi_objective_fitness = MultiFitness(consider_cost=True)
# fitness_weights = (-1., -1.)
# multi_objective_fitness = MultiFitness(consider_resources=True)
# fitness_weights = (-1., -1.)
# multi_objective_fitness = MultiFitness(consider_cost=True, consider_resources=True)
# fitness_weights = (-1., -1., -1.)
# multi_objective_genetic_scheduler_with_estimator = GeneticScheduler(number_of_generation=100, size_of_population=50,
#                                                                     mutate_order=0.05,
#                                                                     mutate_resources=0.005,
#                                                                     sgs_type=ScheduleGenerationScheme.Parallel,
#                                                                     only_lft_initialization=True,
#                                                                     work_estimator=project_work_estimator,
#                                                                     fitness_constructor=multi_objective_fitness,
#                                                                     fitness_weights=fitness_weights,
#                                                                     is_multiobjective=True
#                                                                     )
#
# scheduling_projects = scheduling_pipeline.schedule(multi_objective_genetic_scheduler_with_estimator).finish()
#
# raw_project_schedules = [project.schedule for project in scheduling_projects]
#
# for schedule in raw_project_schedules:
#     # print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}, resources: {count_resources(schedule)}")
#     print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}")


# project_schedule = raw_project_schedule.merged_stages_datetime_df('2022-09-01')
#
# schedule_fig = schedule_gant_chart_fig(schedule_dataframe=project_schedule,
#                                        visualization=VisualizationMode.ShowFig,
#                                        remove_service_tasks=False)
