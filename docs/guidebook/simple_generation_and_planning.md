# Простая генерация и планирование в SAMPO

## Оглавление

* [1. Генерация графа](#1-генерация-графа)

  * [1.1 Простой граф](#11-простой-граф)
  * [1.2 Сложный граф](#12-сложный-граф)
* [2. Генерация подрядчика](#2-генерация-подрядчика)

  * [2.1 Ручная](#21-ручная)
  * [2.2 Из графа](#22-из-графа)
* [3. Планирование](#3-планирование)

  * [3.1 Конструкция планировщика](#31-конструкция-планировщика)
  * [3.2 Процесс планирования](#32-процесс-планирования)
* [4. Метрики для GeneticScheduler](#4-метрики-для-geneticscheduler)

  * [4.1 DeadlineResourcesFitness](#41-deadlineresourcesfitness)
  * [4.2 DeadlineCostFitness](#42-deadlinecostfitness)
  * [4.3 TimeWithResourcesFitness](#43-timewithresourcesfitness)

---

## 1. Генерация графа

### 1.1 Простой граф

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType

r_seed = 231
ss = SimpleSynthetic(r_seed)

# простой граф: 10 кластеров, от 100 до 200 работ в каждом
simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)
```

### 1.2 Сложный граф

```python
# сложный граф: до 2000 работ, 300 уникальных типов работ, 100 типов ресурсов
advanced_wg = ss.advanced_work_graph(
    works_count_top_border=2000,
    uniq_works=300,
    uniq_resources=100,
)
```

---

## 2. Генерация подрядчика

### 2.1 Ручная

```python
from uuid import uuid4
from sampo.schemas.resources import Worker
from sampo.schemas.contractor import Contractor

contractors = [
    Contractor(
        id=str(uuid4()),
        name="OOO Berezka",
        workers={"worker": Worker(id="0", name="worker", count=100)},
    )
]
```

### 2.2 Из графа

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg

contractors = [get_contractor_by_wg(simple_wg)]
```

---

## 3. Планирование

### 3.1 Конструкция планировщика

```python
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.genetic.base import GeneticScheduler

# эвристика HEFT
scheduler = HEFTScheduler()

# или генетический планировщик с простыми гиперпараметрами
scheduler = GeneticScheduler(mutate_order=0.05, mutate_resources=0.05)
```

### 3.2 Процесс планирования

```python
from sampo.pipeline import SchedulingPipeline

project = (
    SchedulingPipeline.create()
    .wg(simple_wg)
    .contractors(contractors)
    .schedule(scheduler)
    .finish()[0]
)

project.schedule.execution_time
```

---

## 4. Метрики для GeneticScheduler

### 4.1 DeadlineResourcesFitness

Оптимизация использования ресурсов при заданном дедлайне.

```python
from sampo.schemas.time import Time
from sampo.scheduler.genetic.operators import DeadlineResourcesFitness
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.pipeline import SchedulingPipeline

deadline = Time(2000)
fitness_constructor = DeadlineResourcesFitness(deadline)

scheduler = GeneticScheduler(
    mutate_order=0.05,
    mutate_resources=0.05,
    fitness_constructor=fitness_constructor,
)
scheduler.set_deadline(deadline)

project = (
    SchedulingPipeline.create()
    .wg(simple_wg)
    .contractors(contractors)
    .schedule(scheduler)
    .finish()[0]
)

project.schedule.execution_time
```

### 4.2 DeadlineCostFitness

Оптимизация стоимости с учётом дедлайна.

```python
from sampo.scheduler.genetic.operators import DeadlineCostFitness
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.pipeline import SchedulingPipeline

fitness_constructor = DeadlineCostFitness(deadline)

scheduler = GeneticScheduler(
    mutate_order=0.05,
    mutate_resources=0.05,
    fitness_constructor=fitness_constructor,
)
scheduler.set_deadline(deadline)

project = (
    SchedulingPipeline.create()
    .wg(simple_wg)
    .contractors(contractors)
    .schedule(scheduler)
    .finish()[0]
)

project.schedule.execution_time
```

### 4.3 TimeWithResourcesFitness

Минимизация времени с учётом ресурсов, без явного дедлайна.

```python
from sampo.scheduler.genetic.operators import TimeWithResourcesFitness
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.pipeline import SchedulingPipeline

fitness_constructor = TimeWithResourcesFitness()

scheduler = GeneticScheduler(
    mutate_order=0.05,
    mutate_resources=0.05,
    fitness_constructor=fitness_constructor,
)
scheduler.set_deadline(deadline)  # при необходимости

project = (
    SchedulingPipeline.create()
    .wg(simple_wg)
    .contractors(contractors)
    .schedule(scheduler)
    .finish()[0]
)

project.schedule.execution_time
```
