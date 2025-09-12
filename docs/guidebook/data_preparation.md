# Подготовка данных для планировщика SAMPO

## Оглавление

* [1. Генерация графа](#1-генерация-графа)

  * [1.1 Синтетические графы](#11-синтетические-графы)
  * [1.2 Сохранение/загрузка WorkGraph](#12-сохранениезагрузка-workgraph)
* [2. Генерация подрядчиков](#2-генерация-подрядчиков)

  * [2.1 Ручная генерация подрядчика](#21-ручная-генерация-подрядчика)
  * [2.2 Синтетическая генерация подрядчиков](#22-синтетическая-генерация-подрядчиков)
  * [2.3 Генерация подрядчика из графа](#23-генерация-подрядчика-из-графа)
  * [2.4 Сохранение/загрузка Contractor](#24-сохранениезагрузка-contractor)

## 1. Генерация графа

### 1.1 Синтетические графы

* Импорт: `SimpleSynthetic`, `SyntheticGraphType`.
* Воспроизводимость: фиксируем зерно `r_seed`.
* Базовый граф: кластерная структура с ограничениями на число работ.
* Продвинутый граф: задаём верхние границы по числу работ и уникальных сущностей.

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType

r_seed = 231
ss = SimpleSynthetic(r_seed)

# Базовый синтетический граф
simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)

# Продвинутый синтетический граф
adv_wg = ss.advanced_work_graph(
    works_count_top_border=2000,
    uniq_works=300,
    uniq_resources=100,
)
```

### 1.2 Сохранение/загрузка WorkGraph

* Сохранение: `WorkGraph.dump(dir, name)` → `name.json`.
* Загрузка: `WorkGraph.load(dir, name)`.
* Проверка идентичности по числу вершин.

```python
from sampo.schemas.graph import WorkGraph

simple_wg.dump(".", "wg")
loaded_simple_wg = WorkGraph.load(".", "wg")
assert simple_wg.vertex_count == loaded_simple_wg.vertex_count
```

## 2. Генерация подрядчиков

### 2.1 Ручная генерация подрядчика

* Импорт: `Contractor`, `Worker`, `uuid4`.
* Заполняем имя и набор ресурсов с количествами.

```python
from uuid import uuid4
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

manual_contractor = Contractor(
    id=str(uuid4()),
    name="OOO Berezka",
    workers={
        "builder": Worker(id=str(uuid4()), name="builder", count=100),
    },
)
```

### 2.2 Синтетическая генерация подрядчиков

* Быстрая генерация тестовых подрядчиков по масштабу ресурсообеспечения.

```python
c5 = ss.contractor(5)
c10 = ss.contractor(10)
c15 = ss.contractor(15)
```

### 2.3 Генерация подрядчика из графа

* Генерация покрытия потребностей конкретного графа.

```python
from sampo.generator.environment import get_contractor_by_wg

contractors = [get_contractor_by_wg(simple_wg)]
```

### 2.4 Сохранение/загрузка Contractor

* Сохранение: `Contractor.dump(dir, name)` → `name.json`.
* Загрузка: `Contractor.load(dir, name)`.

```python
contractors[0].dump(".", "contractor")

from sampo.schemas.contractor import Contractor
loaded_contractor = Contractor.load(".", "contractor")
```
