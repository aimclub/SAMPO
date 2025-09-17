import random
from typing import Optional, List, Tuple, Hashable, Any

from sampo.schemas.graph import GraphNode, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.works import WorkUnit
from sampo.utilities.sampler.requirements import get_worker_reqs_list
from sampo.utilities.sampler.types import MinMax
from sampo.utilities.sampler.works import get_work_unit, get_similar_work_unit


class Sampler:
    def __init__(self,
                 seed: Hashable | None = None):
        self.rand = random.Random(seed)

    def worker_reqs(self,
                    volume: MinMax[int] = MinMax[int](1, 50),
                    worker_count: MinMax[int] = MinMax[int](1, 100)) -> List[WorkerReq]:
        return get_worker_reqs_list(self.rand, volume, worker_count)

    def work_unit(self,
                  model_name: dict[str, Any] | str,
                  work_id: str = '',
                  group: str = 'default',
                  work_volume: MinMax[float] = MinMax[float](0.1, 100.0),
                  req_volume: MinMax[int] = MinMax[int](1, 50),
                  req_worker_count: MinMax[int] = MinMax[int](1, 100)) -> WorkUnit:
        return get_work_unit(self.rand, model_name, work_id, group, work_volume, req_volume, req_worker_count)

    def similar_work_unit(self,
                          exemplar: WorkUnit,
                          scalar: float = 1.0,
                          model_name: dict[str, Any] | str = '',
                          work_id: str = '') -> WorkUnit:
        return get_similar_work_unit(self.rand, exemplar, scalar, model_name, work_id)

    def graph_node(self,
                   model_name: dict[str, Any] | str,
                   edges: list[tuple[GraphNode, float, EdgeType]],
                   work_id: str = '',
                   group: str = 'default',
                   work_volume: MinMax[float] = MinMax[float](0.1, 100.0),
                   req_volume: MinMax[int] = MinMax[int](1, 50),
                   req_worker_count: MinMax[int] = MinMax[int](1, 100)) -> GraphNode:
        wu = get_work_unit(self.rand, model_name, work_id, group, work_volume, req_volume, req_worker_count)
        return GraphNode(wu, edges)

    def similar_graph_node(self,
                           exemplar: GraphNode,
                           edges: list[tuple[GraphNode, float, EdgeType]],
                           scalar: float = 1.0,
                           model_name: dict[str, Any] | str = '',
                           work_id: str = '') -> GraphNode:
        wu = get_similar_work_unit(self.rand, exemplar.work_unit, scalar, model_name, work_id)
        return GraphNode(wu, edges)
