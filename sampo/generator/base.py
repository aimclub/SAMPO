from random import Random

from sampo.generator.environment.contractor import get_contractor
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.generator.pipeline.project import get_small_graph, get_graph
from sampo.generator.pipeline.types import SyntheticGraphType
from sampo.schemas.graph import WorkGraph


class SimpleSynthetic:
    def __init__(self, rand: int | Random | None = None) -> None:
        if isinstance(rand, Random):
            self._rand = rand
        else:
            self._rand = Random(rand)

    def small_work_graph(self, cluster_name: str | None = 'C1') -> WorkGraph:
        """
        Creates a small graph of works consisting of 30-50 vertices;
        :param cluster_name: str - the first cluster name
        :return:
            work_graph: WorkGraph - work graph where count of vertex between 30 and 50
        """
        return get_small_graph(cluster_name, self._rand)

    def work_graph(self, mode: SyntheticGraphType | None = SyntheticGraphType.General,
                   cluster_counts: int | None = 0,
                   bottom_border: int | None = 0,
                   top_border: int | None = 0) -> WorkGraph:
        """
        Invokes a graph of the given type if at least one positive value of
            cluster_counts, bottom_border or top_border is given;
        :param mode: str - 'general' or 'sequence' or 'parallel - the type of the returned graph
        :param cluster_counts: Optional[int] - Number of clusters for the graph
        :param bottom_border: Optional[int] - bottom border for number of works for the graph
        :param top_border: Optional[int] - top border for number of works for the graph
        :return:
            work_graph: WorkGraph - the desired work graph
        """
        return get_graph(mode=mode, cluster_counts=cluster_counts, bottom_border=bottom_border, top_border=top_border,
                         rand=self._rand)

    def contractor(self, pack_worker_count: float):
        """
        Generates a contractor by pack_worker_count and from sampo.generator.environment.contractor.get_contractor
        with default optional parameters
        :param pack_worker_count: The number of resource sets
        :return: the contractor
        """
        return get_contractor(pack_worker_count, rand=self._rand)

    def advanced_work_graph(self, works_count_top_border: int, uniq_works: int, uniq_resources: int):
        """
        Invokes a graph of the given type for given works_count_top_border,
        expands the number of unique resources and job titles, if possible
        :param uniq_resources: Number of unique resources
        :param uniq_works: Number of unique work names
        :param works_count_top_border: Optional[int] - top border for number of works for the graph

        :return: the desired work graph
        """
        wg = self.work_graph(top_border=works_count_top_border)
        wg = extend_names(uniq_works, wg, self._rand)
        wg = extend_resources(uniq_resources, wg, self._rand)
        return wg
