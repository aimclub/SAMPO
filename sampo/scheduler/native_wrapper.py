from deap.base import Toolbox

from sampo.schemas.scheduled_work import ScheduledWork

native = True
try:
    from native import decodeEvaluationInfo
    from native import evaluate as evaluator
    from native import freeEvaluationInfo
except ImportError:
    print("Can't find native module; switching to default")
    decodeEvaluationInfo = lambda *args: args
    freeEvaluationInfo = lambda *args: args
    evaluator = None
    native = False


from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections_util import reverse_dictionary


class NativeWrapper:
    def __init__(self, toolbox: Toolbox, wg: WorkGraph, contractors: list[Contractor], worker_name2index: dict[str, int],
                 worker_pool_indices: dict[int, dict[int, Worker]], time_estimator: WorkTimeEstimator):
        # the outer numeration. Begins with inseparable heads, continuous with tails.
        numeration: dict[int, GraphNode] = {i: node for i, node in
                                            enumerate(filter(lambda node: not node.is_inseparable_son(), wg.nodes))}
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
        self._current_chromosomes = None

        if not native:
            def fit(chromosome: ChromosomeType) -> int:
                sworks = toolbox.chromosome_to_schedule(chromosome)[0]
                return max([swork.finish_time for swork in sworks.values()]).value
            self.evaluator = lambda _, chromosomes: [fit(chromosome) for chromosome in chromosomes]
        else:
            self.evaluator = evaluator

        # preparing C++ cache
        self._cache = decodeEvaluationInfo(self, self.parents, self.inseparables, self.workers, self.totalWorksCount)

    def calculate_working_time(self, chromosome_ind: int, team_target: int, work: int) -> int:
        team = self._current_chromosomes[chromosome_ind][1][team_target]
        workers = [self.worker_pool_indices[worker_index][team[len(self.workers[0])]]
                   .copy().with_count(team[worker_index])
                   for worker_index in range(len(self.workers[0]))
                   if team[worker_index] > 0]
        return self.numeration[work].work_unit.estimate_static(workers, self.time_estimator).value

    def evaluate(self, chromosomes: list[ChromosomeType]):
        self._current_chromosomes = chromosomes
        return self.evaluator(self._cache, chromosomes)

    def close(self):
        freeEvaluationInfo(self._cache)
