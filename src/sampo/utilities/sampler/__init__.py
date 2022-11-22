import random
from typing import Optional, List, Tuple, Hashable

from sampo.schemas.graph import GraphNode, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.works import WorkUnit
from sampo.utilities.sampler.requirements import get_worker_reqs_list
from sampo.utilities.sampler.types import MinMax
from sampo.utilities.sampler.works import get_work_unit, get_similar_work_unit


class Sampler:
    def __init__(self,
                 seed: Optional[Hashable] = None
                 ):
        self.rand = random.Random(seed)

    def worker_reqs(self,
                    volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                    worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                    ) -> List[WorkerReq]:
        return get_worker_reqs_list(self.rand, volume, worker_count)

    def work_unit(self,
                  name: str,
                  work_id: Optional[str] = '',
                  volume_type: Optional[str] = 'unit',
                  group: Optional[str] = "default",
                  work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
                  req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                  req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                  ) -> WorkUnit:
        return get_work_unit(self.rand, name, work_id, volume_type, group, work_volume, req_volume, req_worker_count)

    def similar_work_unit(self,
                          exemplar: WorkUnit,
                          scalar: Optional[float] = 1.0,
                          name: Optional[str] = '',
                          work_id: Optional[str] = ''
                          ) -> WorkUnit:
        return get_similar_work_unit(self.rand, exemplar, scalar, name, work_id)

    def graph_node(self,
                   name: str,
                   edges: List[Tuple[GraphNode, float, EdgeType]],
                   work_id: Optional[str] = '',
                   volume_type: Optional[str] = 'unit',
                   group: Optional[str] = "default",
                   work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
                   req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
                   req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100)
                   ) -> GraphNode:
        wu = get_work_unit(self.rand, name, work_id, volume_type, group, work_volume, req_volume, req_worker_count)
        return GraphNode(wu, edges)

    def similar_graph_node(self,
                           exemplar: GraphNode,
                           edges: List[Tuple[GraphNode, float, EdgeType]],
                           scalar: Optional[float] = 1.0,
                           name: Optional[str] = '',
                           work_id: Optional[str] = ''
                           ) -> GraphNode:
        wu = get_similar_work_unit(self.rand, exemplar.work_unit, scalar, name, work_id)
        return GraphNode(wu, edges)
