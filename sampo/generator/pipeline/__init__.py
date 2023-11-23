from enum import Enum, auto
from sampo.schemas import GraphNode

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


from sampo.generator.pipeline.cluster import get_cluster_works
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.generator.pipeline.project import get_small_graph, get_graph, get_cluster_works
