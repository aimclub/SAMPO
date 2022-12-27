from enum import Enum, auto

from sampo.schemas.graph import GraphNode

StageType = tuple[GraphNode, dict[str, GraphNode]]


class SyntheticGraphType(Enum):
    General = auto()
    Parallel = auto()
    Sequential = auto()
