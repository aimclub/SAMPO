import numpy as np

from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary

from sampo.native import evaluate


class NativeWrapper:
    def __init__(self, wg: WorkGraph, contractors: list[Contractor], worker_name2index: dict[str, int],
                 worker_pool_indices: dict[int, dict[int, Worker]], time_estimator: WorkTimeEstimator):
        # the outer numeration. Begins with inseparable heads, continuous with tails.
        numeration: dict[int, GraphNode] = {i: node for i, node in enumerate(filter(lambda node: not node.is_inseparable_son(), wg.nodes))}
        heads_count = len(numeration)
        for i, node in enumerate([node for node in wg.nodes if node.is_inseparable_son()]):
            numeration[heads_count + i] = node
        rev_numeration = reverse_dictionary(numeration)

        self.numeration = numeration
        # for each vertex index store list of parents' indices
        self.parents = [[rev_numeration[p] for p in numeration[index].parents] for index in range(wg.vertex_count)]
        # for each vertex index store list of whole it's inseparable chain indices
        self.inseparables = [[rev_numeration[p] for p in numeration[index].get_inseparable_chain_with_self()]
                             for index in range(wg.vertex_count)]
        # contractors' workers matrix. If contractor can't supply given type of worker, 0 should be passed
        self.workers = [[0 for _ in range(len(worker_name2index))] for _ in contractors]
        for i, contractor in enumerate(contractors):
            for worker in contractor.workers.values():
                self.workers[i][worker_name2index[worker.name]] = worker.count

        self.totalWorksCount = wg.vertex_count
        self.time_estimator = time_estimator
        self.worker_pool_indices = worker_pool_indices

    def calculate_working_time(self, work: int, contractor: int, team: np.ndarray) -> int:
        workers = [self.worker_pool_indices[worker_index][contractor]
                         .copy().with_count(worker_count)
                         for worker_index, worker_count in enumerate(team)
                         if worker_count > 0]
        return self.numeration[work].work_unit.estimate_static(workers, self.time_estimator).value

    def evaluate(self, chromosomes: list[ChromosomeType]):
        evaluate(self, self.parents, self.inseparables, self.workers, chromosomes)
