# Подготовка данных для планировщика SAMPO

Этот документ описывает способы генерации графов и подрядчиков для тестирования и работы планировщика **SAMPO**.

## Оглавление

* [1. Генерация графа](#1-генерация-графа)

  * [1.1 Синтетические графы](#11-синтетические-графы)
  * [1.2 Сохранение и загрузка WorkGraph](#12-сохранение-и-загрузка-workgraph)
* [2. Генерация подрядчиков](#2-генерация-подрядчиков)

  * [2.1 Ручная генерация подрядчика](#21-ручная-генерация-подрядчика)
  * [2.2 Синтетическая генерация подрядчиков](#22-синтетическая-генерация-подрядчиков)
  * [2.3 Генерация подрядчика из графа](#23-генерация-подрядчика-из-графа)
  * [2.4 Сохранение и загрузка Contractor](#24-сохранение-и-загрузка-contractor)

---

## 1. Генерация графа

### 1.1 Синтетические графы

Импортируем генератор и типы графов:

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType

r_seed = 231  # фиксируем зерно для воспроизводимости
ss = SimpleSynthetic(r_seed)
```

**Варианты графов:**

* **Базовый граф** — кластерная структура с ограничениями на число работ:

```python
simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)
```

* **Продвинутый граф** — с верхними границами по числу работ и уникальных сущностей:

```python
adv_wg = ss.advanced_work_graph(
    works_count_top_border=2000,
    uniq_works=300,
    uniq_resources=100,
)
```

[к оглавлению](#оглавление)

---

### 1.2 Сохранение и загрузка WorkGraph

Формат JSON. Методы одинаковы для любых объектов SAMPO.

* `dump(dir, name)` → сохраняет объект в `name.json`.
* `load(dir, name)` → **устаревший метод** (deprecated), рекомендуется использовать `loadf`.

Пример:

```python
from sampo.schemas.graph import WorkGraph

# Сохраняем граф
simple_wg.dump(".", "wg")

# Загружаем граф (устаревший метод, используйте loadf)
loaded_simple_wg = WorkGraph.load(".", "wg")

# Проверяем идентичность
assert simple_wg.vertex_count == loaded_simple_wg.vertex_count
```

[к оглавлению](#оглавление)

---

## 2. Генерация подрядчиков

### 2.1 Ручная генерация подрядчика

Создание подрядчика с нужными ресурсами вручную:

```python
from uuid import uuid4
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

manual_contractor = Contractor(
    id=str(uuid4()),
    name="OOO Berezka",
    workers={
        "builder": Worker(id=str(uuid4()), name="builder", count=100),  # 100 строителей
    },
)
```

[к оглавлению](#оглавление)

---

### 2.2 Синтетическая генерация подрядчиков

Быстрая генерация тестовых подрядчиков с разным масштабом ресурсов:

```python
c5 = ss.contractor(5)   # подрядчик с малым числом ресурсов
c10 = ss.contractor(10) # средний масштаб
c15 = ss.contractor(15) # крупный масштаб
```

[к оглавлению](#оглавление)

---

### 2.3 Генерация подрядчика из графа

Автоматическое покрытие потребностей конкретного графа:

```python
from sampo.generator.environment import get_contractor_by_wg

contractors = [get_contractor_by_wg(simple_wg)]
```

[к оглавлению](#оглавление)

---

### 2.4 Сохранение и загрузка Contractor

Аналогично WorkGraph:

* `Contractor.dump(dir, name)` → сохраняет объект в `name.json`.
* `Contractor.load(dir, name)` → **устаревший метод**, рекомендуется использовать `loadf`.

Пример:

```python
# Сохраняем подрядчика
contractors[0].dump(".", "contractor")

# Загружаем подрядчика (устаревший метод)
from sampo.schemas.contractor import Contractor
loaded_contractor = Contractor.load(".", "contractor")
```

[к оглавлению](#оглавление)
