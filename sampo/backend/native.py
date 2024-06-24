from random import Random

from native import decodeEvaluationInfo, evaluate, evaluate_to_schedules

from sampo.api.genetic_api import FitnessFunction, ChromosomeType, ScheduleGenerationScheme
from sampo.backend.default import DefaultComputationalBackend
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.schemas import Time, Schedule, GraphNode, WorkGraph, Contractor, LandscapeConfiguration, WorkTimeEstimator, \
    Worker, ScheduledWork
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.utilities.collections_util import reverse_dictionary


class NativeComputationalBackend(DefaultComputationalBackend):

    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec, rand: Random | None = None,
                             work_estimator: WorkTimeEstimator | None = None):
        super().cache_scheduler_info(wg, contractors, landscape, spec, rand, work_estimator)

        self._cache = decodeEvaluationInfo(self, wg, contractors)

    def cache_genetic_info(self, population_size: int, mutate_order: float, mutate_resources: float,
                           mutate_zones: float, deadline: Time | None, weights: list[int] | None,
                           init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                           assigned_parent_time: Time, fitness_weights: tuple[int | float, ...],
                           sgs_type: ScheduleGenerationScheme, only_lft_initialization: bool, is_multiobjective: bool):
        # TODO
        return super().cache_genetic_info(population_size, mutate_order, mutate_resources, mutate_zones, deadline,
                                          weights, init_schedules, assigned_parent_time, fitness_weights, sgs_type,
                                          only_lft_initialization, is_multiobjective)

    def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
        return evaluate(self._cache, chromosomes)

    def compute_chromosomes_to_schedules(self, chromosomes: list[ChromosomeType]) -> list[Schedule]:
        return [self.get_schedule(summary) for summary in evaluate_to_schedules(self._cache, chromosomes)]

    def get_schedule(self, summary: list[tuple[str, int, int, list[tuple[str, str, int, str]], str]]) -> Schedule:
        works = [ScheduledWork(work_unit=self._wg[work_id].work_unit,
                               start_end_time=(Time(start_time_val), Time(finish_time_val)),
                               workers=[Worker(id=worker_id,
                                               name=name,
                                               count=count,
                                               contractor_id=contractor_id)
                                        for worker_id, name, count, contractor_id in workers],
                               contractor=contractor_name)
                 for work_id, start_time_val, finish_time_val, workers, contractor_name in summary]
        return Schedule.from_scheduled_works(works, self._wg)

    # def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
    #     return evaluate(self._cache, chromosomes, self._mutate_order, self._mutate_resources, self._mutate_resources,
    #                     self._mutate_order, self._mutate_resources, self._mutate_resources, 50)
