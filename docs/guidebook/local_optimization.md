# Локальная оптимизация в SAMPO

Документ описывает использование локальных оптимизаторов для улучшения расписаний в **SAMPO**.

## Оглавление

* [1. Подготовка данных](#1-подготовка-данных)

  * [1.1 Генерация графа и ресурсов](#11-генерация-графа-и-ресурсов)
  * [1.2 Планировщик](#12-планировщик)
* [2. Локальная оптимизация](#2-локальная-оптимизация)

  * [2.1 Оптимизация порядка (Scheduling order)](#21-оптимизация-порядка-scheduling-order)
  * [2.2 Оптимизация расписания (Schedule)](#22-оптимизация-расписания-schedule)
  * [2.3 Комбинированное применение](#23-комбинированное-применение)

---

## 1. Подготовка данных

### 1.1 Генерация графа и ресурсов

Используем синтетический генератор для построения графа и формируем подрядчика на основе его потребностей.
Чтобы избежать ошибок при локальной оптимизации, рекомендуется зафиксировать `id` подрядчика и использовать один и тот же объект на всех шагах.

```python
from uuid import uuid5, NAMESPACE_DNS
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg

r_seed = 231
ss = SimpleSynthetic(r_seed)

simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)

# Генерация подрядчика и фиксация стабильного id
contractor = get_contractor_by_wg(simple_wg)
contractor.id = str(uuid5(NAMESPACE_DNS, "contractor-by-wg"))
contractors = [contractor]
```

[к оглавлению](#оглавление)

---

### 1.2 Планировщик

Для построения расписания используется планировщик и конвейер:

* базовый планировщик: `HEFTScheduler`;
* конвейер: `SchedulingPipeline`.

```python
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.pipeline import SchedulingPipeline

scheduler = HEFTScheduler()
```

[к оглавлению](#оглавление)

---

## 2. Локальная оптимизация

В SAMPO доступны два типа локальной оптимизации:

* **Order** — перестановка порядка планирования работ;
* **Schedule** — перерасчёт частей расписания с использованием альтернативных таймлайнов.

---

### 2.1 Оптимизация порядка (Scheduling order)

Класс: `SwapOrderLocalOptimizer`.
Применяется к диапазону вершин графа до вызова `.schedule(...)`.

```python
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer

order_optimizer = SwapOrderLocalOptimizer()

project = SchedulingPipeline.create() \
    .wg(simple_wg) \
    .contractors(contractors) \
    .optimize_local(order_optimizer, range(0, 10)) \
    .schedule(scheduler) \
    .finish()[0]

project.schedule.execution_time
```

[к оглавлению](#оглавление)

---

### 2.2 Оптимизация расписания (Schedule)

Класс: `ParallelizeScheduleLocalOptimizer`.
Требует выбора таймлайна, например `JustInTimeTimeline`.
Обычно применяется после построения базового расписания.
Важно: необходимо использовать один и тот же объект подрядчика с фиксированным `id`.

```python
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.utils.local_optimization import ParallelizeScheduleLocalOptimizer

schedule_optimizer = ParallelizeScheduleLocalOptimizer(JustInTimeTimeline)

project = SchedulingPipeline.create() \
    .wg(simple_wg) \
    .contractors(contractors) \
    .schedule(scheduler) \
    .optimize_local(schedule_optimizer, range(0, 5)) \
    .finish()[0]

project.schedule.execution_time
```

[к оглавлению](#оглавление)

---

### 2.3 Комбинированное применение

Можно комбинировать несколько оптимизаторов.
Пример: сначала оптимизация порядка на двух частях графа, затем оптимизация расписания на этих же диапазонах.
Следите, чтобы подрядчик был один и тот же, без пересоздания.

```python
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer, ParallelizeScheduleLocalOptimizer
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline

order_optimizer = SwapOrderLocalOptimizer()
schedule_optimizer = ParallelizeScheduleLocalOptimizer(JustInTimeTimeline)

half = simple_wg.vertex_count // 2

project = SchedulingPipeline.create() \
    .wg(simple_wg) \
    .contractors(contractors) \
    .optimize_local(order_optimizer, range(0, half)) \
    .optimize_local(order_optimizer, range(half, simple_wg.vertex_count)) \
    .schedule(scheduler) \
    .optimize_local(schedule_optimizer, range(0, half)) \
    .optimize_local(schedule_optimizer, range(half, simple_wg.vertex_count)) \
    .finish()[0]

project.schedule.execution_time
```

[к оглавлению](#оглавление)

---

## Примечания

* Если при оптимизации появляется ошибка `KeyError` по `contractor_id`, причина обычно в том, что подрядчик пересоздан или его `id` изменился.
* Для стабильной работы используйте один и тот же объект подрядчика и фиксируйте его `id`.
