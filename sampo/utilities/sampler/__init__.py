"""Sampling utilities for generating random scheduling entities.

Sampling utilities for generating random scheduling entities.
Утилиты выборки для генерации случайных сущностей расписания.
"""

import random
from typing import Optional, List, Tuple, Hashable

from sampo.schemas.graph import GraphNode, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.works import WorkUnit
from sampo.utilities.sampler.requirements import get_worker_reqs_list
from sampo.utilities.sampler.types import MinMax
from sampo.utilities.sampler.works import get_work_unit, get_similar_work_unit


class Sampler:
    """Generates random work units and graph nodes.

    Generates random work units and graph nodes.
    Генерирует случайные рабочие единицы и узлы графа.
    """

    def __init__(self, seed: Optional[Hashable] = None) -> None:
        """Create sampler with optional random seed.

        Create sampler with optional random seed.
        Создает объект выборки с необязательным случайным зерном.

        Args:
            seed: Base for random generator. seed: Основа для генератора
                случайных чисел.
        """

        self.rand = random.Random(seed)

    def worker_reqs(
        self,
        volume: Optional[MinMax[int]] = MinMax[int](1, 50),
        worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
    ) -> List[WorkerReq]:
        """Generate random worker requirements.

        Generate random worker requirements.
        Сгенерировать случайные требования к рабочим.

        Args:
            volume: Range of total work volume. volume: Диапазон общего
                объема работ.
            worker_count: Range of worker numbers. worker_count: Диапазон
                количества рабочих.

        Returns:
            List[WorkerReq]: Worker requirements list. List[WorkerReq]:
                Список требований к рабочим.
        """

        return get_worker_reqs_list(self.rand, volume, worker_count)

    def work_unit(
        self,
        name: str,
        work_id: Optional[str] = "",
        volume_type: Optional[str] = "unit",
        group: Optional[str] = "default",
        work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
        req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
        req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
    ) -> WorkUnit:
        """Create a random work unit.

        Create a random work unit.
        Создать случайную рабочую единицу.

        Args:
            name: Work unit name. name: Название рабочей единицы.
            work_id: Identifier of work unit. work_id: Идентификатор
                рабочей единицы.
            volume_type: Unit of volume. volume_type: Единица измерения
                объема.
            group: Group for generated work. group: Группа для
                сгенерированной работы.
            work_volume: Range for work volume. work_volume: Диапазон
                объема работ.
            req_volume: Range for requirement volume. req_volume:
                Диапазон объема требований.
            req_worker_count: Range of workers per requirement.
                req_worker_count: Диапазон числа рабочих на требование.

        Returns:
            WorkUnit: Generated work unit. WorkUnit: Сгенерированная
                рабочая единица.
        """

        return get_work_unit(
            self.rand,
            name,
            work_id,
            volume_type,
            group,
            work_volume,
            req_volume,
            req_worker_count,
        )

    def similar_work_unit(
        self,
        exemplar: WorkUnit,
        scalar: Optional[float] = 1.0,
        name: Optional[str] = "",
        work_id: Optional[str] = "",
    ) -> WorkUnit:
        """Create work unit similar to exemplar.

        Create work unit similar to exemplar.
        Создать рабочую единицу, похожую на образец.

        Args:
            exemplar: Source work unit. exemplar: Исходная рабочая
                единица.
            scalar: Scale factor for volume. scalar: Коэффициент масштабирования
                объема.
            name: New name if provided. name: Новое имя, если указано.
            work_id: New identifier if provided. work_id: Новый
                идентификатор, если указан.

        Returns:
            WorkUnit: Generated similar work unit. WorkUnit:
                Сгенерированная похожая рабочая единица.
        """

        return get_similar_work_unit(self.rand, exemplar, scalar, name, work_id)

    def graph_node(
        self,
        name: str,
        edges: List[Tuple[GraphNode, float, EdgeType]],
        work_id: Optional[str] = "",
        volume_type: Optional[str] = "unit",
        group: Optional[str] = "default",
        work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
        req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
        req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
    ) -> GraphNode:
        """Create graph node with random work unit.

        Create graph node with random work unit.
        Создать узел графа со случайной рабочей единицей.

        Args:
            name: Node name. name: Название узла.
            edges: Connections to other nodes. edges: Связи с другими
                узлами.
            work_id: Identifier of work unit. work_id: Идентификатор
                рабочей единицы.
            volume_type: Unit of volume. volume_type: Единица измерения
                объема.
            group: Group for generated work. group: Группа
                сгенерированной работы.
            work_volume: Range for work volume. work_volume: Диапазон
                объема работ.
            req_volume: Range for requirement volume. req_volume:
                Диапазон объема требований.
            req_worker_count: Range of workers per requirement.
                req_worker_count: Диапазон числа рабочих на требование.

        Returns:
            GraphNode: Generated graph node. GraphNode: Сгенерированный
                узел графа.
        """

        wu = get_work_unit(
            self.rand,
            name,
            work_id,
            volume_type,
            group,
            work_volume,
            req_volume,
            req_worker_count,
        )
        return GraphNode(wu, edges)

    def similar_graph_node(
        self,
        exemplar: GraphNode,
        edges: List[Tuple[GraphNode, float, EdgeType]],
        scalar: Optional[float] = 1.0,
        name: Optional[str] = "",
        work_id: Optional[str] = "",
    ) -> GraphNode:
        """Create node similar to exemplar.

        Create node similar to exemplar.
        Создать узел, похожий на образец.

        Args:
            exemplar: Source node. exemplar: Исходный узел.
            edges: Connections to other nodes. edges: Связи с другими
                узлами.
            scalar: Scale factor for volume. scalar: Коэффициент
                масштабирования объема.
            name: New name if provided. name: Новое имя, если указано.
            work_id: New identifier if provided. work_id: Новый
                идентификатор, если указан.

        Returns:
            GraphNode: Generated similar node. GraphNode: Сгенерированный
                похожий узел.
        """

        wu = get_similar_work_unit(self.rand, exemplar.work_unit, scalar, name, work_id)
        return GraphNode(wu, edges)
