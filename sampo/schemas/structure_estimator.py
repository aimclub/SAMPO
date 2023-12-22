from abc import ABC, abstractmethod

from sampo.schemas import GraphNode, WorkGraph


class StructureProbabilityEstimator(ABC):
    @abstractmethod
    def get_generate_probability(self, node: GraphNode) -> dict[str, float]:
        ...


class StructureEstimator(ABC):

    @abstractmethod
    def restruct(self, wg: WorkGraph) -> WorkGraph:
        ...
