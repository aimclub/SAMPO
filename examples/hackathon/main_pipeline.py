import json
from datetime import datetime
import warnings
from typing import Optional
import os

from sampo.scheduler.genetic import ScheduleGenerationScheme, TimeFitness
from sampo.scheduler import GeneticScheduler, HEFTScheduler


from prepare_pipeline import WorkEstimatorType, get_pipeline_with_estimator

from xml_parser import convert_dates_in_schedule, process_schedule, schedule_csv_to_xml, get_project_works_structure

from skills_resource_optimizer import GreedyMinimalMultiSkillResourceOptimizer

from fitness import MultiFitness, WeightedFitness

from tuning import tune_genetic

warnings.filterwarnings("ignore")


def single_scheduling(data_path: str, output_path: Optional[str] = None,
                      time_weight: float = 0.5, cost_weight: float = 0.5, resources_weight: float = 0.,
                      estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                      working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)):
    """
    Это функция для однокритериальной оптимизации расписания
    Возвращает найденное оптимальное расписание
    """
    scheduling_pipeline, project_work_estimator = get_pipeline_with_estimator(data_path, estimator_type, working_hours,
                                                                              start_date)
    if abs(time_weight) + abs(cost_weight) + abs(resources_weight) == 0:
        raise Exception('At least one weight should be non-zero')  # нельзя занулять все критерии
    if time_weight == 0:
        # если вес критерия времени равен нулю, то оптимизация тривиальна
        # - берем всегда одних и тех же самых дешевых работников
        scheduler = HEFTScheduler(resource_optimizer=GreedyMinimalMultiSkillResourceOptimizer(),
                                  work_estimator=project_work_estimator)
    else:
        if abs(cost_weight) + abs(resources_weight) == 0:
            # если веса остальных критериев равны нулю, то оптимизируем чисто по времени
            fitness = TimeFitness()
        else:
            # в остальных случаях суммируем критерии с соответствующими весами
            # - и оптимизируем по этой взвешенной сумме
            fitness = WeightedFitness(time_weight=time_weight, cost_weight=cost_weight,
                                      resources_weight=resources_weight)
        if not os.path.exists('best_params.json'):
            # если затьюненные параметры не сохранены, то запускаем тьюнинг и сохраняем их
            tune_genetic(data_path, 1)

        with open('best_params.json', 'r') as f:
            # загружаем затьюненные параметры
            params = json.load(f)
            mutate_order = params['mut_order_pb']
            mutate_resources = params['mut_res_pb']

        scheduler = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                     mutate_order=mutate_order,
                                     mutate_resources=mutate_resources,
                                     sgs_type=ScheduleGenerationScheme.Parallel,
                                     only_lft_initialization=True,
                                     work_estimator=project_work_estimator,
                                     fitness_constructor=fitness,
                                     )
    scheduling_project = scheduling_pipeline.schedule(scheduler).finish()[0]

    schedule = scheduling_project.schedule
    schedule_df = convert_dates_in_schedule(schedule.pure_schedule_df, start_date)

    if output_path is not None:
        schedule_df.to_csv(os.path.join(output_path, 'schedule.csv'), index=0)
        _, *structure_info = get_project_works_structure(data_path)
        schedule_csv_to_xml(*process_schedule(schedule_df, structure_info),
                            data_path, os.path.join(output_path, 'schedule.xml'))

    return schedule


def multi_scheduling(data_path: str, output_path: Optional[str] = None,
                     consider_cost: bool = True, consider_resources: bool = False,
                     estimator_type: WorkEstimatorType = WorkEstimatorType.Calendar,
                     working_hours: int = 8, start_date: datetime = datetime(2024, 2, 5)):
    """
    Это функция для мультикритериальной оптимизации расписания
    Возвращает найденные парето-оптимальные расписания
    """
    if not consider_cost and not consider_resources:
        # если оба дополнительных критерия не указаны, то это не мультикритериальная задача,
        # а однокритериальная по времени
        raise Exception('At least one additional criteria should be considered')
    scheduling_pipeline, project_work_estimator = get_pipeline_with_estimator(data_path, estimator_type, working_hours,
                                                                              start_date)
    # задаем критерий оптимизации, возвращающий кортеж интересующих критериев: (время, [стоимость], [кол-во ресурсов])
    fitness = MultiFitness(consider_cost=consider_cost, consider_resources=consider_resources)
    # задаем веса для критериев. нас интересует минимизация, поэтому веса отрицательные
    weights = (-1., -1.)
    if consider_cost and consider_resources:
        weights = (*weights, -1.)
    if not os.path.exists('best_params.json'):
        # если затьюненные параметры не сохранены, то запускаем тьюнинг и сохраняем их
        tune_genetic(data_path, 1)

    with open('best_params.json', 'r') as f:
        # загружаем затьюненные параметры
        params = json.load(f)
        mutate_order = params['mut_order_pb']
        mutate_resources = params['mut_res_pb']

    scheduler = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                 mutate_order=mutate_order,
                                 mutate_resources=mutate_resources,
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
            schedule_df = convert_dates_in_schedule(schedule.pure_schedule_df, start_date)
            schedule_df.to_csv(os.path.join(output_path, f'schedule_{i}.csv'), index=0)

            _, *structure_info = get_project_works_structure(data_path)
            schedule_csv_to_xml(*process_schedule(schedule_df, structure_info),
                                data_path,
                                os.path.join(output_path, f'schedule_{i}.xml'))
    return schedules


if __name__ == "__main__":
    from fitness import count_resources

    filepath = './sber_task.xml'

    raw_project_schedule = single_scheduling(filepath)

    print(raw_project_schedule.execution_time.value)

    # raw_project_schedules = multi_scheduling(filepath)
    #
    # for schedule in raw_project_schedules:
    #     # print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}, resources: {count_resources(schedule)}")
    #     print(f"time: {schedule.execution_time.value}, cost: {schedule.pure_schedule_df['cost'].sum()}")
