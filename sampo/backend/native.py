from random import Random

from native import decodeEvaluationInfo, evaluate

from sampo.api.genetic_api import FitnessFunction, ChromosomeType, ScheduleGenerationScheme
from sampo.backend.default import DefaultComputationalBackend
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.schemas import Time, Schedule, GraphNode, WorkGraph, Contractor, LandscapeConfiguration, WorkTimeEstimator
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
        # the outer numeration. Begins with inseparable heads, continuous with tails.

        worker_pool = get_worker_contractor_pool(contractors)

        index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(wg.nodes)}
        work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
        worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}

        numeration: dict[int, GraphNode] = {i: node for i, node in
                                            enumerate(filter(lambda node: not node.is_inseparable_son(), wg.nodes))}
        heads_count = len(numeration)
        for i, node in enumerate([node for node in wg.nodes if node.is_inseparable_son()]):
            numeration[heads_count + i] = node
        rev_numeration = reverse_dictionary(numeration)

        # TODO remove assignment unuseful info to self

        # for each vertex index store list of parents' indices
        parents = [[rev_numeration[p] for p in numeration[index].parents] for index in range(wg.vertex_count)]
        head_parents = [list(parents[i]) for i in range(len(parents))]
        # for each vertex index store list of whole it's inseparable chain indices
        inseparables = [[rev_numeration[p] for p in numeration[index].get_inseparable_chain_with_self()]
                             for index in range(wg.vertex_count)]
        # contractors' workers matrix. If contractor can't supply given type of worker, 0 should be passed
        workers = [[0 for _ in range(len(worker_name2index))] for _ in contractors]
        for i, contractor in enumerate(contractors):
            for worker in contractor.workers.values():
                workers[i][worker_name2index[worker.name]] = worker.count

        min_req = [[] for _ in range(len(numeration))]  # np.zeros((len(numeration), len(worker_pool_indices)))
        max_req = [[] for _ in range(len(numeration))]  # np.zeros((len(numeration), len(worker_pool_indices)))
        for work_index, node in numeration.items():
            cur_min_req = [0 for _ in worker_name2index]
            cur_max_req = [0 for _ in worker_name2index]
            for req in node.work_unit.worker_reqs:
                worker_index = worker_name2index[req.kind]
                cur_min_req[worker_index] = req.min_count
                cur_max_req[worker_index] = req.max_count
            min_req[work_index] = cur_min_req
            max_req[work_index] = cur_max_req

        volume = [node.work_unit.volume for node in numeration.values()]

        id2work = [numeration[i].work_unit.id for i in range(len(numeration))]
        id2worker_name = reverse_dictionary(worker_name2index)
        id2res = [id2worker_name[i] for i in range(len(id2worker_name))]

        # from native import ddd
        #
        # ddd(wg)

        self._cache = decodeEvaluationInfo(self, wg, contractors, "", parents, head_parents, inseparables, workers,
                                           wg.vertex_count, False, True, volume, min_req, max_req, id2work, id2res)

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

    # def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
    #     return evaluate(self._cache, chromosomes, self._mutate_order, self._mutate_resources, self._mutate_resources,
    #                     self._mutate_order, self._mutate_resources, self._mutate_resources, 50)
