import random
from typing import Optional, Callable

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.genetic.operators import FitnessFunction, TimeFitness
from sampo.scheduler.genetic.schedule_builder import build_schedule
from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.identity import IdentityResourceOptimizer
from sampo.scheduler.resources_in_time.average_binary_search import AverageBinarySearchResourceOptimizingScheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.validation import validate_schedule


class GeneticScheduler(Scheduler):
    """
    Class for hybrid scheduling algorithm, that uses heuristic algorithm to generate
    first population and genetic algorithm to search the best solving
    """

    def __init__(self,
                 number_of_generation: Optional[int] = 50,
                 mutate_order: Optional[float or None] = None,
                 mutate_resources: Optional[float or None] = None,
                 mutate_zones: Optional[float or None] = None,
                 size_of_population: Optional[float or None] = None,
                 rand: Optional[random.Random] = None,
                 seed: Optional[float or None] = None,
                 n_cpu: int = 1,
                 weights: list[int] = None,
                 fitness_constructor: Callable[[Callable[[list[ChromosomeType]], list[Schedule]]], FitnessFunction] = TimeFitness,
                 scheduler_type: SchedulerType = SchedulerType.Genetic,
                 resource_optimizer: ResourceOptimizer = IdentityResourceOptimizer(),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 optimize_resources: bool = False,
                 verbose: bool = True):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=resource_optimizer,
                         work_estimator=work_estimator)
        self.number_of_generation = number_of_generation
        self.mutate_order = mutate_order
        self.mutate_resources = mutate_resources
        self.mutate_zones = mutate_zones
        self.size_of_population = size_of_population
        self.rand = rand or random.Random(seed)
        self.fitness_constructor = fitness_constructor
        self.work_estimator = work_estimator
        self._optimize_resources = optimize_resources
        self._n_cpu = n_cpu
        self._weights = weights
        self._verbose = verbose

        self._time_border = None
        self._deadline = None

    def __str__(self) -> str:
        return f'GeneticScheduler[' \
               f'generations={self.number_of_generation},' \
               f'population_size={self.size_of_population},' \
               f'mutate_order={self.mutate_order},' \
               f'mutate_resources={self.mutate_resources}' \
               f']'

    def get_params(self, works_count: int) -> tuple[float, float, float, int]:
        """
        Return base parameters for model to make new population

        :param works_count:
        :return:
        """
        mutate_order = self.mutate_order
        if mutate_order is None:
            mutate_order = 0.05

        mutate_resources = self.mutate_resources
        if mutate_resources is None:
            mutate_resources = 0.005

        mutate_zones = self.mutate_zones
        if mutate_zones is None:
            mutate_zones = 0.05

        size_of_population = self.size_of_population
        if size_of_population is None:
            if works_count < 300:
                size_of_population = 50
            elif 1500 > works_count >= 300:
                size_of_population = 100
            else:
                size_of_population = works_count // 25
        return mutate_order, mutate_resources, mutate_zones, size_of_population

    def set_use_multiprocessing(self, n_cpu: int):
        """
        Set the number of CPU cores.
        DEPRECATED, NOT WORKING

        :param n_cpu:
        """
        self._n_cpu = n_cpu

    def set_time_border(self, time_border: int):
        self._time_border = time_border

    def set_deadline(self, deadline: Time):
        """
        Set the deadline of tasks

        :param deadline:
        """
        self._deadline = deadline

    def set_weights(self, weights: list[int]):
        self._weights = weights

    def set_optimize_resources(self, optimize_resources: bool):
        self._optimize_resources = optimize_resources

    def set_verbose(self, verbose: bool):
        self._verbose = verbose

    @staticmethod
    def generate_first_population(wg: WorkGraph, contractors: list[Contractor],
                                  landscape: LandscapeConfiguration = LandscapeConfiguration(),
                                  spec: ScheduleSpec = ScheduleSpec(),
                                  work_estimator: WorkTimeEstimator = None,
                                  deadline: Time = None,
                                  weights=None):
        """
        Algorithm, that generate first population

        :param landscape:
        :param wg: graph of works
        :param contractors:
        :param spec:
        :return:
        """

        if weights is None:
            weights = [2, 2, 1, 1, 1, 1]

        def init_k_schedule(scheduler_class, k):
            try:
                return scheduler_class(work_estimator=work_estimator,
                                       resource_optimizer=AverageReqResourceOptimizer(k)) \
                    .schedule(wg, contractors,
                              spec,
                              landscape=landscape), list(reversed(prioritization(wg, work_estimator))), spec
            except NoSufficientContractorError:
                return None, None, None

        if deadline is None:
            def init_schedule(scheduler_class):
                try:
                    return scheduler_class(work_estimator=work_estimator).schedule(wg, contractors,
                                                                                   landscape=landscape), \
                        list(reversed(prioritization(wg, work_estimator))), spec
                except NoSufficientContractorError:
                    return None, None, None

        else:
            def init_schedule(scheduler_class):
                try:
                    (schedule, _, _, _), modified_spec = AverageBinarySearchResourceOptimizingScheduler(
                        scheduler_class(work_estimator=work_estimator)
                    ).schedule_with_cache(wg, contractors, deadline, spec, landscape=landscape)
                    return schedule, list(reversed(prioritization(wg, work_estimator))), modified_spec
                except NoSufficientContractorError:
                    return None, None, None

        return {
            "heft_end": (*init_schedule(HEFTScheduler), weights[0]),
            "heft_between": (*init_schedule(HEFTBetweenScheduler), weights[1]),
            "12.5%": (*init_k_schedule(HEFTScheduler, 8), weights[2]),
            "25%": (*init_k_schedule(HEFTScheduler, 4), weights[3]),
            "75%": (*init_k_schedule(HEFTScheduler, 4 / 3), weights[4]),
            "87.5%": (*init_k_schedule(HEFTScheduler, 8 / 7), weights[5])
        }

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            landscape: LandscapeConfiguration = LandscapeConfiguration(),
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
        """
        Build schedule for received graph of workers and return the current state of schedule
        It's needed to use this method in multy agents model

        :param landscape:
        :param wg:
        :param contractors:
        :param spec:
        :param validate:
        :param assigned_parent_time:
        :param timeline:
        :return:
        """
        init_schedules = GeneticScheduler.generate_first_population(wg, contractors, landscape, spec,
                                                                    self.work_estimator, self._deadline, self._weights)

        mutate_order, mutate_resources, mutate_zones, size_of_population = self.get_params(wg.vertex_count)
        worker_pool = get_worker_contractor_pool(contractors)
        deadline = None if self._optimize_resources else self._deadline

        scheduled_works, schedule_start_time, timeline, order_nodes = build_schedule(wg,
                                                                                     contractors,
                                                                                     worker_pool,
                                                                                     size_of_population,
                                                                                     self.number_of_generation,
                                                                                     mutate_order,
                                                                                     mutate_resources,
                                                                                     mutate_zones,
                                                                                     init_schedules,
                                                                                     self.rand,
                                                                                     spec,
                                                                                     landscape,
                                                                                     self.fitness_constructor,
                                                                                     self.work_estimator,
                                                                                     self._n_cpu,
                                                                                     assigned_parent_time,
                                                                                     timeline,
                                                                                     self._time_border,
                                                                                     self._optimize_resources,
                                                                                     deadline,
                                                                                     self._verbose)
        schedule = Schedule.from_scheduled_works(scheduled_works.values(), wg)

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline, order_nodes
