# Проект планирования в SAMPO (SchedulingProject)

## Оглавление

* [1. Генерация графа](#1-генерация-графа)

  * [1.1 Простой граф](#11-простой-граф)
  * [1.2 Сложный граф](#12-сложный-граф)
* [2. Генерация подрядчика](#2-генерация-подрядчика)

  * [2.1 Ручная](#21-ручная)
  * [2.2 Из графа](#22-из-графа)
* [3. Планирование](#3-планирование)

  * [3.1 Конструкция планировщика](#31-конструкция-планировщика)
  * [3.2 Расчёт через SchedulingPipeline](#32-расчёт-через-schedulingpipeline)
  * [3.3 Сериализация проекта](#33-сериализация-проекта)

## 1. Генерация графа

### 1.1 Простой граф

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType

r_seed = 231
ss = SimpleSynthetic(r_seed)

simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)
```

### 1.2 Сложный граф

```python
advanced_wg = ss.advanced_work_graph(
    works_count_top_border=2000,
    uniq_works=300,
    uniq_resources=100,
)
```

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

## 3. Планирование

### 3.1 Конструкция планировщика

```python
from sampo.scheduler.heft.base import HEFTScheduler

scheduler = HEFTScheduler()
```

### 3.2 Расчёт через SchedulingPipeline

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

### 3.3 Сериализация проекта

```python
project.dump(".", "project_test")
```
