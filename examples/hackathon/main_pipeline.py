from datetime import datetime
import warnings
from typing import Optional
import os

from sampo.pipeline import SchedulingPipeline, InputPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler import GeneticScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme, TimeFitness
from sampo.schemas import WorkTimeEstimator

from work_time_estimator import WorkEstimator, CalendarBasedWorkEstimator

from xml_parser import get_works_info, get_contractors_info, get_project_calendar

from fitness import MultiFitness, WeightedFitness

from enum import Enum

warnings.filterwarnings("ignore")


class WorkEstimatorType(Enum):
    Basic = 'Basic'
    Calendar = 'Calendar'


def get_pipeline_with_estimator(data_path: str, estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                                working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)) \
        -> tuple[InputPipeline, WorkTimeEstimator]:
    df = get_works_info(data_path)
    contractors = get_contractors_info(data_path)

    scheduling_pipeline = SchedulingPipeline.create()
    scheduling_pipeline = scheduling_pipeline.wg(wg=df,
                                                 all_connections=True,
                                                 change_connections_info=False)

    scheduling_pipeline = scheduling_pipeline.contractors(contractors).lag_optimize(LagOptimizationStrategy.TRUE)

    match estimator_type:
        case WorkEstimatorType.Basic:
            project_work_estimator = WorkEstimator()
        case WorkEstimatorType.Calendar:
            project_business_calendar = get_project_calendar(data_path)
            project_work_estimator = CalendarBasedWorkEstimator(project_calendar=project_business_calendar,
                                                                working_hours_cnt=working_hours,
                                                                start_date=start_date)
        case _:
            raise Exception('Invalid WorkEstimator type')

    scheduling_pipeline = scheduling_pipeline.work_estimator(project_work_estimator)
    return scheduling_pipeline, project_work_estimator


def single_scheduling(data_path: str, output_path: Optional[str] = None,
                      time_weight: float = 0.5, cost_weight: float = 0.5, resources_weight: float = 0.,
                      estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                      working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)):
    scheduling_pipeline, project_work_estimator = get_pipeline_with_estimator(data_path, estimator_type, working_hours,
                                                                              start_date)
    if cost_weight + resources_weight == 0:
        fitness = TimeFitness()
    else:
        fitness = WeightedFitness(time_weight=time_weight, cost_weight=cost_weight, resources_weight=resources_weight)
    scheduler = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                 mutate_order=0.05,
                                 mutate_resources=0.005,
                                 sgs_type=ScheduleGenerationScheme.Parallel,
                                 only_lft_initialization=True,
                                 work_estimator=project_work_estimator,
                                 fitness_constructor=fitness,
                                 )
    scheduling_project = scheduling_pipeline.schedule(scheduler).finish()[0]

    schedule = scheduling_project.schedule

    if output_path is not None:
        raw_project_schedule.pure_schedule_df.to_csv(output_path, index=0)
    return schedule


def multi_scheduling(data_path: str, output_path: Optional[str] = None,
                     consider_cost: bool = True, consider_resources: bool = False,
                     estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                     working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)):
    if not consider_cost and not consider_resources:
        raise Exception('At least one additional criteria should be considered')
    scheduling_pipeline, project_work_estimator = get_pipeline_with_estimator(data_path, estimator_type, working_hours,
                                                                              start_date)
    fitness = MultiFitness(consider_cost=consider_cost, consider_resources=consider_resources)
    weights = (-1., -1.)
    if consider_cost and consider_resources:
        weights = (*weights, -1.)
    scheduler = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                 mutate_order=0.05,
                                 mutate_resources=0.005,
                                 sgs_type=ScheduleGenerationScheme.Parallel,
                                 only_lft_initialization=True,
                                 work_estimator=project_work_estimator,
                                 fitness_constructor=fitness,
                                 fitness_weights=weights,
                                 is_multiobjective=True
                                 )

    scheduling_projects = scheduling_pipeline.schedule(scheduler).finish()

    schedules = [project.schedule for project in scheduling_projects]

    if output_path is not None:
        for i, schedule in enumerate(schedules):
            schedule.pure_schedule_df.to_csv(os.path.join(output_path, f'schedule_{i}.csv'), index=0)
    return schedules


if __name__ == "__main__":
    from fitness import count_resources

    filepath = './sber_task.xml'

    raw_project_schedule = single_scheduling(filepath, 'scheduled.csv')

    print(raw_project_schedule.execution_time.value)

    raw_project_schedules = multi_scheduling(filepath)

    for schedule in raw_project_schedules:
        # print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}, resources: {count_resources(schedule)}")
        print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}")
