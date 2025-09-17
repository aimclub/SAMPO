# Оценщик времени и ресурсов FieldDevWorkEstimator

## Оглавление

* [1. Назначение и зависимости](#1-назначение-и-зависимости)
* [2. Константы и инициализация модели](#2-константы-и-инициализация-модели)
* [3. Публичные методы](#3-публичные-методы)

  * [3.1 `estimate_time`](#31-estimate_time)
  * [3.2 `find_work_resources`](#32-find_work_resources)
  * [3.3 `set_estimation_mode`](#33-set_estimation_mode)
  * [3.4 `set_productivity_mode`](#34-set_productivity_mode)
  * [3.5 `get_recreate_info`](#35-get_recreate_info)
* [4. Пример использования](#4-пример-использования)

---

## 1. Назначение и зависимости

`FieldDevWorkEstimator` — пользовательский `WorkTimeEstimator` для SAMPO. Возвращает длительности работ и требования к ресурсам на базе внешней модели `ResTimeModel`, подключённой через адаптер `MschmAdapter`.

```python
import logging
from random import Random
from typing import Type
from itertools import chain
from operator import attrgetter

from sampo.schemas.time import Time
from sampo.schemas import (
    WorkTimeEstimator, WorkUnit, Worker, WorkerReq,
    WorkEstimationMode, WorkerProductivityMode
)

from idbadapter import MschmAdapter
from stairsres.res_time_model import ResTimeModel
from sampo.utilities.collections_util import build_index
```

---

## 2. Константы и инициализация модели

Сервисные работы имеют нулевую длительность. Модель создаётся один раз и переиспользуется. Логгер ведёт предупреждения по сбоям оценки.

```python
SERVICE_WORKS = [
    "Начало работ по марке", "Окончание работ по марке",
    "NaN", "start of project", "finish of project"
]

URL = "test"
model = ResTimeModel(MschmAdapter(url=URL))
logger = logging.getLogger("field-dev-estimator-log")


class FieldDevWorkEstimator(WorkTimeEstimator):
    def __init__(self, rand: Random = Random()):
        self._url = URL
        self._model = model
        self._use_idle = True
        self._estimation_mode = WorkEstimationMode.Realistic
        self.rand = rand
        self._productivity_mode = WorkerProductivityMode.Static
```

---

## 3. Публичные методы

### 3.1 `estimate_time`

Логика:

* Имя работы очищается от суффикса `_stage_…`.
* Список ресурсов превращается в `[{name, _count}]`; недостающие из `worker_reqs` добавляются с `_count=0`.
* Режимы маппятся на квантили `"0.1" | "0.5" | "0.9"`.
* Сервисные работы → `Time(0)`.
* Ошибки логируются и не прерывают процесс.

```python
def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
    w_u = {
        "name": work_unit.name.split("_stage_")[0],
        "volume": work_unit.volume,
        "measurement": work_unit.volume_type,
    }

    w_l = [{"name": w.name, "_count": w.count} for w in worker_list]
    name2worker = build_index(worker_list, attrgetter("name"))

    match self._estimation_mode:
        case WorkEstimationMode.Optimistic:
            mode_str = "0.1"
        case WorkEstimationMode.Realistic:
            mode_str = "0.5"
        case _:
            mode_str = "0.9"

    # добавляем отсутствующие ресурсы с нулевым количеством
    for req in work_unit.worker_reqs:
        if name2worker.get(req.kind) is None:
            w_l.append({"name": req.kind, "_count": 0})

    if w_u["name"] in SERVICE_WORKS:
        return Time(0)

    try:
        return Time(int(self._model.estimate_time(work_unit=w_u, worker_list=w_l, mode=mode_str)))
    except Exception as e:
        logger.warning(f"Couldn't estimate time for work unit with name='{w_u['name']}': {e}")
        # допустимо вернуть 0 или эвристическое значение; здесь вернём 0
        return Time(0)
```

### 3.2 `find_work_resources`

Возвращает плоский список `WorkerReq`, рассчитанный моделью.

```python
def find_work_resources(
    self,
    work_name: str,
    work_volume: float,
    resource_name: list[str] | None = None,
    measurement: str | None = None
) -> list[WorkerReq]:
    if work_name in SERVICE_WORKS:
        return []

    worker_req_dict = self._model.get_resources_volumes(
        work_name=work_name,
        work_volume=work_volume,
        measurement=measurement
    ) or {}

    worker_reqs = [
        [
            WorkerReq(
                kind=req["kind"],
                volume=Time(req["volume"]),
                min_count=req["min_count"],
                max_count=req["max_count"],
            )
            for req in req_list
        ]
        for req_list in worker_req_dict.values()
    ]
    return list(chain.from_iterable(worker_reqs))
```

### 3.3 `set_estimation_mode`

```python
def set_estimation_mode(
    self,
    use_idle: bool = True,
    mode: WorkEstimationMode = WorkEstimationMode.Realistic
) -> None:
    self._use_idle = use_idle
    self._estimation_mode = mode
```

### 3.4 `set_productivity_mode`

```python
def set_productivity_mode(
    self,
    mode: WorkerProductivityMode = WorkerProductivityMode.Static
) -> None:
    self._productivity_mode = mode
```

### 3.5 `get_recreate_info`

Возвращает конструктор и «параметры» для восстановления. Сейчас из `_url: str` берётся `tuple(self._url)`, то есть кортеж символов строки.

```python
from typing import Type

def get_recreate_info(self) -> tuple[Type, tuple]:
    return FieldDevWorkEstimator, (self._url,) # "test"
```

---

## 4. Пример использования

```python
from random import Random
from sampo.schemas import WorkUnit, Worker
from sampo.schemas import WorkEstimationMode, WorkerProductivityMode

est = FieldDevWorkEstimator(rand=Random(231))

# оценка времени
wu = WorkUnit(name="Бурение_stage_1", volume=100.0, volume_type="м")
workers = [Worker(id="1", name="буровик", count=5)]
t = est.estimate_time(wu, workers)

# требования к ресурсам
reqs = est.find_work_resources("Бурение", 100.0, measurement="м")

# переключение режимов
est.set_estimation_mode(use_idle=True, mode=WorkEstimationMode.Realistic)
est.set_productivity_mode(mode=WorkerProductivityMode.Static)
```

Минимальная настройка логирования:

```python
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
```