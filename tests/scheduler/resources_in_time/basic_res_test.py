from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.genetic.operators import DeadlineResourcesFitness
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.resources_in_time.average_binary_search import AverageBinarySearchResourceOptimizingScheduler
from sampo.schemas.time import Time
from sampo.utilities.resource_cost import schedule_cost


def test_deadline_planning(setup_wg, setup_contractors):
    scheduler = AverageBinarySearchResourceOptimizingScheduler(HEFTScheduler())

    deadline = Time(30)

    schedule, _, _, _ = scheduler.schedule_with_cache(setup_wg, setup_contractors, deadline)

    print(f'Planning for deadline time: {schedule.execution_time}, cost: {schedule_cost(schedule)}')

    scheduler = HEFTScheduler()

    schedule, _, _, _ = scheduler.schedule_with_cache(setup_wg, setup_contractors)

    print(f'Plain planning time: {schedule.execution_time}, cost: {schedule_cost(schedule)}')


def test_genetic_deadline_planning(setup_wg, setup_contractors):
    deadline = Time.inf() // 2
    scheduler = GeneticScheduler(fitness_constructor=DeadlineResourcesFitness.prepare(deadline))
    scheduler.set_deadline(deadline)

    schedule = scheduler.schedule(setup_wg, setup_contractors)

    print(f'Planning for deadline time: {schedule.execution_time}, cost: {schedule_cost(schedule)}')
