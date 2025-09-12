# Локальная оптимизация в SAMPO

## Оглавление

* [1. Подготовка данных](#1-подготовка-данных)

  * [1.1 Генерация графа и ресурсов](#11-генерация-графа-и-ресурсов)
  * [1.2 Планировщик](#12-планировщик)
* [2. Локальная оптимизация](#2-локальная-оптимизация)

  * [2.1 Оптимизация порядка (Scheduling order)](#21-оптимизация-порядка-scheduling-order)
  * [2.2 Оптимизация расписания (Schedule)](#22-оптимизация-расписания-schedule)
  * [2.3 Комбинированное применение](#23-комбинированное-применение)

## 1. Подготовка данных

### 1.1 Генерация графа и ресурсов

* Используем синтетический генератор и подрядчика из графа.
* Фиксируем `r_seed` для воспроизводимости.

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg

r_seed = 231
ss = SimpleSynthetic(r_seed)

simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)

contractors = [get_contractor_by_wg(simple_wg)]
```

### 1.2 Планировщик

* Базовый планировщик: `HEFTScheduler`.
* Конвейер: `SchedulingPipeline`.

```python
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.pipeline import SchedulingPipeline

scheduler = HEFTScheduler()
```

## 2. Локальная оптимизация

В SAMPO два вида локальной оптимизации:

* **Order** — перестановка порядка планирования работ для улучшения результата.
* **Schedule** — перерасчёт частей расписания для уменьшения времени выполнения.

### 2.1 Оптимизация порядка (Scheduling order)

* Класс: `SwapOrderLocalOptimizer`.
* Применение к поддиапазону вершин графа через `.optimize_local(...)` до вызова `.schedule(...)`.

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

### 2.2 Оптимизация расписания (Schedule)

* Класс: `ParallelizeScheduleLocalOptimizer`.
* Требует выбор таймлайна, например `JustInTimeTimeline`.
* Обычно применяется после базового расписания.

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

### 2.3 Комбинированное применение

* Можно стекать несколько локальных оптимизаторов.
* Пример: сначала оптимизация порядка на разных диапазонах, затем расписания.

```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.utils.local_optimization import SwapOrderLocalOptimizer, ParallelizeScheduleLocalOptimizer
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline

order_optimizer = SwapOrderLocalOptimizer()
schedule_optimizer = ParallelizeScheduleLocalOptimizer(JustInTimeTimeline)

project = SchedulingPipeline.create() \
    .wg(simple_wg) \
    .contractors(contractors) \
    .optimize_local(order_optimizer, range(0, simple_wg.vertex_count // 2)) \
    .optimize_local(order_optimizer, range(simple_wg.vertex_count // 2, simple_wg.vertex_count)) \
    .schedule(scheduler) \
    .optimize_local(schedule_optimizer, range(0, simple_wg.vertex_count // 2)) \
    .optimize_local(schedule_optimizer, range(simple_wg.vertex_count // 2, simple_wg.vertex_count)) \
    .finish()[0]

project.schedule.execution_time
```
