from enum import Enum, auto

from sampo.schemas.graph import GraphNode

StageType = tuple[GraphNode, dict[str, GraphNode]]


class SyntheticGraphType(Enum):
    """
    Describe available types of synthetic graph

    * PARALLEL - work graph dominated by parallel works
    * SEQUENTIAL - work graph dominated by sequential works
    * GENERAL - work graph, including sequential and parallel works, it is similar to the real work graphs
    """
    GENERAL = auto()
    PARALLEL = auto()
    SEQUENTIAL = auto()
