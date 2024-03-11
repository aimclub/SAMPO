from sampo.scheduler.genetic import GeneticScheduler, TimeAndResourcesFitness, ScheduleGenerationScheme
from sampo.utilities.resource_usage import resources_peaks_sum


def test_multiobjective_genetic_scheduling(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    genetic = GeneticScheduler(number_of_generation=10,
                               mutate_order=0.05,
                               mutate_resources=0.05,
                               size_of_population=20,
                               fitness_constructor=TimeAndResourcesFitness(),
                               fitness_weights=(-1, -1),
                               optimize_resources=True,
                               is_multiobjective=True,
                               sgs_type=ScheduleGenerationScheme.Serial)

    schedules = genetic.schedule(setup_wg, setup_contractors, validate=True, landscape=setup_landscape)
    assert isinstance(schedules, list) and len(schedules)
    fitnesses = [(schedule.execution_time.value, resources_peaks_sum(schedule)) for schedule in schedules]
    print('\nPareto-efficient fitnesses:\n', fitnesses)
