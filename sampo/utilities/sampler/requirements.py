"""Utilities for generating worker requirements.

Utilities for generating worker requirements.
Утилиты для генерации требований к рабочим.
"""

import random
from typing import Optional

from sampo.schemas.requirements import WorkerReq
from sampo.utilities.sampler.resources import (
    WORKER_TYPES,
    WorkerSpecialization,
)
from sampo.utilities.sampler.types import MinMax


def get_worker_req(
    rand: random.Random,
    name: str,
    volume: Optional[MinMax[int]] = MinMax[int](1, 50),
    worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
) -> WorkerReq:
    """Generate requirement for a single worker type.

    Generate requirement for a single worker type.
    Сгенерировать требование для одного типа рабочего.

    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        name: Worker specialization. name: Специализация рабочего.
        volume: Range of required volume. volume: Диапазон требуемого объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.

    Returns:
        WorkerReq: Requirement description. WorkerReq: Описание
            требования.
    """

    count = rand.randint(volume.min, volume.max)
    return WorkerReq(name, count, worker_count.min, worker_count.max)


def get_worker_reqs_list(
    rand: random.Random,
    volume: Optional[MinMax[int]] = MinMax[int](1, 50),
    worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
) -> list[WorkerReq]:
    """Generate list of random worker requirements.

    Generate list of random worker requirements.
    Сгенерировать список случайных требований к рабочим.

    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        volume: Range of required volume. volume: Диапазон требуемого
            объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.

    Returns:
        list[WorkerReq]: Worker requirements list. list[WorkerReq]: Список
            требований к рабочим.
    """

    names: list[WorkerSpecialization] = list(WORKER_TYPES)
    rand.shuffle(names)
    req_count = rand.randint(1, len(names))
    names = names[:req_count]
    return get_worker_specific_reqs_list(rand, names, volume, worker_count)


def get_worker_specific_reqs_list(
    rand: random.Random,
    worker_names: list[WorkerSpecialization],
    volume: Optional[MinMax[int]] = MinMax[int](1, 50),
    worker_count: Optional[MinMax[int]] = MinMax[int](1, 100),
) -> list[WorkerReq]:
    """Generate requirements for specific worker types.

    Generate requirements for specific worker types.
    Сгенерировать требования для конкретных типов рабочих.

    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        worker_names: Specializations list. worker_names: Список специализаций.
        volume: Range of required volume. volume: Диапазон требуемого
            объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.

    Returns:
        list[WorkerReq]: Worker requirements list. list[WorkerReq]: Список
            требований к рабочим.
    """

    return [get_worker_req(rand, name, volume, worker_count) for name in worker_names]
