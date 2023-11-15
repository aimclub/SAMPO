from enum import Enum, auto

from sampo.schemas.graph import GraphNode

StageType = tuple[GraphNode, dict[str, GraphNode]]


class SyntheticGraphType(Enum):
    """
    Describe types of synthetic graph:
    - Parallel - work graph in which has work is parallel to each other
    - Sequential - work graph in which the work is sequential to each other
    - General - work graph that closely resemble real work graphs in structure.
    """
    GENERAL = auto()
    PARALLEL = auto()
    SEQUENTIAL = auto()
