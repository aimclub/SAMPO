import pytest
import math

from sampo.scheduler.genetic import GeneticScheduler, DeadlineResourcesFitness, SumOfResourcesPeaksFitness
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.resources_in_time.average_binary_search import AverageBinarySearchResourceOptimizingScheduler
from sampo.utilities.resource_usage import resources_costs_sum, resources_peaks_sum
from sampo.schemas.time import Time


def test_deadline_planning(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    scheduler = AverageBinarySearchResourceOptimizingScheduler(HEFTScheduler())

    deadline = Time(30)

    schedule = scheduler.schedule_with_cache(setup_wg, setup_contractors, deadline, landscape=setup_landscape)[0][0]

    if schedule is None:
        pytest.skip("Given contractors can't satisfy given work graph")

    print(f'Planning for deadline time: {schedule.execution_time}, cost: {resources_costs_sum(schedule)}')

    scheduler = HEFTScheduler()

    schedule, _, _, _ = scheduler.schedule_with_cache(setup_wg, setup_contractors, landscape=setup_landscape)

    print(f'Plain planning time: {schedule.execution_time}, cost: {resources_costs_sum(schedule)}')


def test_genetic_deadline_planning(setup_scheduler_parameters):
    setup_wg, setup_contractors, landscape = setup_scheduler_parameters

    deadline = Time.inf() // 2
    scheduler = GeneticScheduler(number_of_generation=5,
                                 mutate_order=0.05,
                                 mutate_resources=0.005,
                                 size_of_population=50,
                                 fitness_constructor=DeadlineResourcesFitness.prepare(deadline),
                                 optimize_resources=True,
                                 verbose=False)

    scheduler.set_deadline(deadline)

    schedule = scheduler.schedule(setup_wg, setup_contractors, landscape=landscape)

    print(f'Planning for deadline time: {schedule.execution_time}, ' +
          f'peaks: {resources_peaks_sum(schedule)}, cost: {resources_costs_sum(schedule)}')


def test_true_deadline_planning(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    scheduler = AverageBinarySearchResourceOptimizingScheduler(HEFTScheduler())

    deadline = Time(5000)
    (deadlined_schedule, _, _, _), _ = scheduler.schedule_with_cache(setup_wg, setup_contractors, deadline,
                                                                     landscape=setup_landscape)

    if deadlined_schedule is None:
        pytest.skip('1 Given contractors cannot satisfy given work graph')
    peak_deadlined = resources_peaks_sum(deadlined_schedule)

    deadline = Time.inf() // 2
    (not_deadlined_schedule, _, _, _), _ = scheduler.schedule_with_cache(setup_wg, setup_contractors, deadline,
                                                                         landscape=setup_landscape)
    if deadlined_schedule is None:
        pytest.skip('2 Given contractors cannot satisfy given work graph')
    peak_not_deadlined = resources_peaks_sum(not_deadlined_schedule)

    print(f'Peak with    deadline: {peak_deadlined}, time: {deadlined_schedule.execution_time}')
    print(f'Peak without deadline: {peak_not_deadlined}, time: {not_deadlined_schedule.execution_time}')


def test_lexicographic_genetic_deadline_planning(setup_scheduler_parameters):
    setup_wg, setup_contractors, setup_landscape = setup_scheduler_parameters

    scheduler = HEFTScheduler()
    schedule, _, _, _ = scheduler.schedule_with_cache(setup_wg, setup_contractors, landscape=setup_landscape)

    # assigning deadline to the time-10^(order_of_magnitude(time) - 1)
    # time - time of schedule from HEFT
    deadline = schedule.execution_time
    deadline -= 10 ** max(0, int(math.log10(deadline.value) - 1))

    print(f'Deadline time: {deadline}')

    scheduler_combined = GeneticScheduler(number_of_generation=5,
                                          mutate_order=0.05,
                                          mutate_resources=0.05,
                                          size_of_population=50,
                                          fitness_constructor=DeadlineResourcesFitness.prepare(deadline),
                                          optimize_resources=True,
                                          verbose=False)

    scheduler_combined.set_deadline(deadline)

    scheduler_lexicographic = GeneticScheduler(number_of_generation=5,
                                               mutate_order=0.05,
                                               mutate_resources=0.05,
                                               size_of_population=50,
                                               fitness_constructor=SumOfResourcesPeaksFitness,
                                               verbose=False)

    scheduler_lexicographic.set_deadline(deadline)

    schedule = scheduler_combined.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
    time_combined = schedule.execution_time

    print(f'\tCombined genetic: time = {time_combined}, ' +
          f'peak = {resources_peaks_sum(schedule)}')

    schedule = scheduler_lexicographic.schedule(setup_wg, setup_contractors, landscape=setup_landscape)
    time_lexicographic = schedule.execution_time

    print(f'\tLexicographic genetic: time = {time_lexicographic}, ' +
          f'peak = {resources_peaks_sum(schedule)}')
    print()
