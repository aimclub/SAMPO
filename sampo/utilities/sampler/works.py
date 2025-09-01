"""Work unit sampling helpers.

Work unit sampling helpers.
Вспомогательные функции выборки рабочих единиц.
"""

import random
from typing import Optional

from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit
from sampo.utilities.sampler.requirements import get_worker_reqs_list
from sampo.utilities.sampler.types import MinMax


def get_work_unit(
    rand: random.Random,
    name: str,
    work_id: Optional[str] = "",
    volume_type: Optional[str] = "unit",
    group: Optional[str] = "default",
    work_volume: Optional[MinMax[float]] = MinMax[float](0.1, 100.0),
    req_volume: Optional[MinMax[int]] = MinMax[int](1, 50),
    req_worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
) -> WorkUnit:
    """Generate a random work unit.

    Generate a random work unit.
    Сгенерировать случайную рабочую единицу.

    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        name: Name of work unit. name: Название рабочей единицы.
        work_id: Identifier of work unit. work_id: Идентификатор
            рабочей единицы.
        volume_type: Unit of volume. volume_type: Единица измерения объема.
        group: Group of work. group: Группа работы.
        work_volume: Range of work volume. work_volume: Диапазон объема
            работ.
        req_volume: Range of requirement volume. req_volume: Диапазон
            объема требований.
        req_worker_count: Range of worker numbers per requirement.
            req_worker_count: Диапазон чисел рабочих на требование.

    Returns:
        WorkUnit: Generated work unit. WorkUnit: Сгенерированная рабочая
            единица.
    """

    reqs = get_worker_reqs_list(rand, req_volume, req_worker_count)
    work_id = work_id or uuid_str(rand)
    volume = rand.random() * (work_volume.max - work_volume.min) + work_volume.min
    return WorkUnit(
        work_id,
        name,
        worker_reqs=reqs,
        volume=volume,
        volume_type=volume_type,
        group=group,
    )


def get_similar_work_unit(
    rand: random.Random,
    exemplar: WorkUnit,
    scalar: Optional[float] = 1.0,
    name: Optional[str] = "",
    work_id: Optional[str] = "",
) -> WorkUnit:
    """Generate work unit similar to exemplar.

    Generate work unit similar to exemplar.
    Сгенерировать рабочую единицу, подобную образцу.

    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        exemplar: Base work unit. exemplar: Базовая рабочая единица.
        scalar: Scale factor for volume. scalar: Коэффициент масштабирования
            объема.
        name: New name if provided. name: Новое имя, если указано.
        work_id: New identifier if provided. work_id: Новый идентификатор,
            если указан.

    Returns:
        WorkUnit: Generated work unit. WorkUnit: Сгенерированная рабочая
            единица.
    """

    reqs = [req.scale_all(scalar) for req in exemplar.worker_reqs]
    work_id = work_id or uuid_str(rand)
    name = name or exemplar.name
    return WorkUnit(
        work_id,
        name,
        worker_reqs=reqs,
        group=exemplar.group,
        volume=exemplar.volume * scalar,
        volume_type=exemplar.volume_type,
    )
