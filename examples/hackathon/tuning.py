import optuna
import json
import warnings

from sampo.scheduler import GeneticScheduler
from sampo.scheduler.genetic import ScheduleGenerationScheme

from main_pipeline import get_pipeline_with_estimator


warnings.filterwarnings("ignore")

filepath = './sber_task.xml'

scheduling_pipeline, project_work_estimator = get_pipeline_with_estimator(filepath)

n_iters = 5


def objective(trial: optuna.Trial):
    mutate_order = trial.suggest_float('mut_order_pb', 0.01, 0.07)
    mutate_resources = trial.suggest_float('mut_res_pb', 0.001, 0.01)

    time = 0

    for step in range(n_iters):
        genetic_scheduler_with_estimator = GeneticScheduler(number_of_generation=100, size_of_population=50,
                                                            mutate_order=mutate_order,
                                                            mutate_resources=mutate_resources,
                                                            sgs_type=ScheduleGenerationScheme.Parallel,
                                                            only_lft_initialization=True,
                                                            work_estimator=project_work_estimator
                                                            )

        scheduling_project = scheduling_pipeline.schedule(genetic_scheduler_with_estimator).finish()[0]
        schedule_time = scheduling_project.schedule.execution_time.value

        time += schedule_time

        trial.report(schedule_time, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return time / n_iters


study = optuna.create_study()
study.optimize(objective, n_trials=20)
print(study.best_params)
with open('best_params.json', 'w') as f:
    json.dump(study.best_params, f)
