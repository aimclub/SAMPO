from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from random import Random
from uuid import uuid4

from sampo.schemas import GraphNode, WorkGraph, WorkUnit, WorkTimeEstimator, IntervalGaussian, Interval
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.structurator.prepare_wg_copy import prepare_work_graph_copy


class StructureGenerationEstimator(ABC):
    @abstractmethod
    def generate_probabilities(self, parent: WorkUnit) -> dict[str, float]:
        ...

    @abstractmethod
    def get_volume(self, parent: WorkUnit, target: str) -> float:
        ...

class DefaultStructureGenerationEstimator(StructureGenerationEstimator):

    def __init__(self, rand: Random | None = None):
        self._gen_probabilities = defaultdict(lambda: defaultdict(float))
        self._rand = rand or Random()
        self._volume = IntervalGaussian(5, 2, 0.1, 10, self._rand)

    def set_probabilities(self, defect_probabilities: dict[str, dict[str, float]]) -> 'DefaultStructureGenerationEstimator':
        self._gen_probabilities = defect_probabilities
        return self

    def set_probability(self, parent: str, child: str, probability: float) -> 'DefaultStructureGenerationEstimator':
        self._gen_probabilities[parent][child] = probability
        return self

    def set_volume_interval(self, volume: Interval) -> 'DefaultStructureGenerationEstimator':
        self._volume = volume
        return self

    def generate_probabilities(self, parent: WorkUnit) -> dict[str, float]:
        return self._gen_probabilities[parent]

    def get_volume(self, parent: WorkUnit, target: str) -> float:
        return self._volume.rand_float()


class StructureEstimator(ABC):

    @abstractmethod
    def restruct(self, wg: WorkGraph) -> WorkGraph:
        ...

class DefaultStructureEstimator(StructureEstimator):

    def __init__(self, generator: StructureGenerationEstimator, rand: Random | None = None):
        self._generator = generator
        self._work_estimator = DefaultWorkEstimator()
        self._rand = rand or Random()

    def set_work_estimator(self, work_estimator: WorkTimeEstimator) -> 'DefaultStructureEstimator':
        self._work_estimator = work_estimator
        return self

    def _construct_sub_work(self, work_unit: WorkUnit, sub_work_name: str) -> WorkUnit:
        zone_reqs = deepcopy(work_unit.zone_reqs)
        volume = float(self._generator.get_volume(work_unit, sub_work_name))

        worker_reqs = self._work_estimator.find_work_resources(sub_work_name, volume)
        return WorkUnit(str(uuid4()), sub_work_name,
                        worker_reqs=worker_reqs, zone_reqs=zone_reqs, volume=volume)

    def _gen_sub_works(self, work_unit: WorkUnit) -> list[WorkUnit]:
        probabilities = list(self._generator.generate_probabilities(work_unit).items())
        return [self._construct_sub_work(work_unit, name)
                for name, prob in probabilities
                if self._rand.random() < prob]

    def restruct(self, wg: WorkGraph) -> WorkGraph:
        nodes, _ = prepare_work_graph_copy(wg, change_id=False)
        start = nodes[wg.start.id]
        finish = nodes[wg.finish.id]
        for i, node in enumerate(nodes.values()):
            if node.work_unit.is_service_unit:
                continue
            for sub_work in self._gen_sub_works(node.work_unit):
                # construct node, add edge to the parent
                sub_node = GraphNode(sub_work, [node])
                # add edge to the children of the routine
                finish.add_parents([sub_node])

        return WorkGraph(start, finish)
