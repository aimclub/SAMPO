"""Helpers for generating synthetic project data.

Вспомогательные функции для генерации синтетических данных проекта.
"""

from random import Random

from sampo.generator import SyntheticGraphType
from sampo.generator.environment import get_contractor
from sampo.generator.environment.landscape import get_landscape_by_wg
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.generator.pipeline.project import get_small_graph, get_graph
from sampo.schemas import LandscapeConfiguration, MaterialReq
from sampo.schemas.graph import WorkGraph


class SimpleSynthetic:
    """Simplified interface for synthetic data generation.

    Упрощенный интерфейс для генерации синтетических данных.
    """

    def __init__(self, rand: int | Random | None = None) -> None:
        """Create generator with optional random seed.

        Создает генератор с необязательным источником случайности.

        Args:
            rand (int | Random | None): Seed or random generator.
                Зерно или генератор случайных чисел.
        """

        if isinstance(rand, Random):
            self._rand = rand
        else:
            self._rand = Random(rand)

    def small_work_graph(self, cluster_name: str | None = 'C1') -> WorkGraph:
        """Create small work graph with 30-50 vertices.

        Создает небольшой граф работ из 30-50 вершин.

        Args:
            cluster_name (str | None): Name of the first cluster.
                Имя первого кластера.

        Returns:
            WorkGraph: Generated work graph.
                Сгенерированный граф работ.
        """

        return get_small_graph(cluster_name, self._rand)

    def work_graph(
        self,
        mode: SyntheticGraphType | None = SyntheticGraphType.GENERAL,
        cluster_counts: int | None = 0,
        bottom_border: int | None = 0,
        top_border: int | None = 0,
    ) -> WorkGraph:
        """Generate work graph of the chosen type.

        Генерирует граф работ выбранного типа.

        Args:
            mode (SyntheticGraphType | None): Graph type.
                Тип графа.
            cluster_counts (int | None): Number of clusters.
                Количество кластеров.
            bottom_border (int | None): Lower bound for works.
                Нижняя граница работ.
            top_border (int | None): Upper bound for works.
                Верхняя граница работ.

        Returns:
            WorkGraph: Generated work graph.
                Сгенерированный граф работ.
        """

        return get_graph(
            mode=mode,
            cluster_counts=cluster_counts,
            bottom_border=bottom_border,
            top_border=top_border,
            rand=self._rand,
        )

    def contractor(self, pack_worker_count: float):
        """Generate contractor with default parameters.

        Генерирует подрядчика с параметрами по умолчанию.

        Args:
            pack_worker_count (float): Number of resource sets.
                Количество наборов ресурсов.

        Returns:
            Contractor: Generated contractor.
                Сгенерированный подрядчик.
        """

        return get_contractor(pack_worker_count, rand=self._rand)

    def advanced_work_graph(
        self, works_count_top_border: int, uniq_works: int, uniq_resources: int
    ) -> WorkGraph:
        """Generate graph and extend names and resources.

        Генерирует граф и расширяет названия и ресурсы.

        Args:
            works_count_top_border (int): Upper bound for work count.
                Верхняя граница числа работ.
            uniq_works (int): Number of unique work names.
                Количество уникальных названий работ.
            uniq_resources (int): Number of unique resources.
                Количество уникальных ресурсов.

        Returns:
            WorkGraph: Generated work graph.
                Сгенерированный граф работ.
        """

        wg = self.work_graph(top_border=works_count_top_border)
        wg = extend_names(uniq_works, wg, self._rand)
        wg = extend_resources(uniq_resources, wg, self._rand)
        return wg

    def set_materials_for_wg(
        self,
        wg: WorkGraph,
        materials_name: list[str] = None,
        bottom_border: int = None,
        top_border: int = None,
    ) -> WorkGraph:
        """Assign materials to work graph nodes.

        Назначает материалы узлам графа работ.

        Args:
            wg (WorkGraph): Work graph to modify.
                Граф работ для изменения.
            materials_name (list[str] | None): Available material names.
                Доступные названия материалов.
            bottom_border (int | None): Minimal kinds per node.
                Минимальное число видов на узел.
            top_border (int | None): Maximal kinds per node.
                Максимальное число видов на узел.

        Returns:
            WorkGraph: Work graph with materials.
                Граф работ с материалами.

        Raises:
            ValueError: Borders exceed materials list.
                Границы превышают список материалов.
        """

        if materials_name is None:
            materials_name = [
                'stone',
                'brick',
                'sand',
                'rubble',
                'concrete',
                'metal',
            ]
            bottom_border = 2
            top_border = 6
        else:
            if bottom_border is None:
                bottom_border = len(materials_name) // 2
            if top_border is None:
                top_border = len(materials_name)

        if bottom_border > len(materials_name) or top_border > len(materials_name):
            raise ValueError('The borders are out of the range of materials_name')

        for node in wg.nodes:
            if not node.work_unit.is_service_unit:
                work_materials = list(
                    set(
                        self._rand.choices(
                            materials_name,
                            k=self._rand.randint(bottom_border, top_border),
                        )
                    )
                )
                node.work_unit.material_reqs = [
                    MaterialReq(name, self._rand.randint(52, 345), name)
                    for name in work_materials
                ]

        return wg

    def synthetic_landscape(self, wg: WorkGraph) -> LandscapeConfiguration:
        """Generate landscape for the given work graph.

        Генерирует ландшафт для заданного графа работ.

        Args:
            wg (WorkGraph): Work graph.
                Граф работ.

        Returns:
            LandscapeConfiguration: Generated landscape.
                Сгенерированный ландшафт.
        """

        return get_landscape_by_wg(wg, self._rand)
