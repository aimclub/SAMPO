from datetime import datetime
from enum import Enum

from sampo.pipeline import SchedulingPipeline, InputPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.schemas import WorkTimeEstimator


from work_time_estimator import WorkEstimator, CalendarBasedWorkEstimator

from xml_parser import get_works_info, get_contractors_info, get_project_calendar


class WorkEstimatorType(Enum):
    Basic = 'Basic'
    Calendar = 'Calendar'


def get_pipeline_with_estimator(data_path: str, estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                                working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)) \
        -> tuple[InputPipeline, WorkTimeEstimator]:
    df, _, _, _ = get_works_info(data_path)
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
