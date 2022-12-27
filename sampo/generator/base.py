from random import Random

from sampo.generator.environment.contractor import get_contractor
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.generator.pipeline.project import get_small_graph, get_graph
from sampo.generator.types import SyntheticGraphType
from sampo.schemas.graph import WorkGraph


class SimpleSynthetic:
    def __init__(self, rand: int | Random | None) -> None:
        if isinstance(rand, Random):
            self._rand = rand
        else:
            self._rand = Random(rand)

    def small_work_graph(self, cluster_name: str | None = 'C1') -> WorkGraph:
        return get_small_graph(cluster_name, self._rand)

    def work_graph(self, mode: SyntheticGraphType | None = SyntheticGraphType.General,
                   bottom_border: int | None = 0,
                   top_border: int | None = 0) -> WorkGraph:
        return get_graph(mode=mode, bottom_border=bottom_border, top_border=top_border, rand=self._rand)

    def contactor(self, pack_worker_count: float):
        return get_contractor(pack_worker_count, rand=self._rand)

    def advanced_work_graph(self, works_count_top_border: int, uniq_works: int, uniq_resources: int, scaler: int = 5):
        wg = self.work_graph(top_border=works_count_top_border)
        wg = extend_names(uniq_works, wg, self._rand)
        wg = extend_resources(uniq_resources, wg, self._rand)
        return wg
