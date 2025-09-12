# Сценарии генерации и модификации графов в SAMPO (generator\_scenarios.py)

## Оглавление

* [1. Инициализация и базовая генерация](#1-инициализация-и-базовая-генерация)
* [2. Базовое планирование HEFT](#2-базовое-планирование-heft)
* [3. Подрядчик по WorkGraph](#3-подрядчик-по-workgraph)
* [4. Расширение номенклатуры работ](#4-расширение-номенклатуры-работ)
* [5. Расширение номенклатуры ресурсов](#5-расширение-номенклатуры-ресурсов)
* [6. Проверка согласованности генерации подрядчика](#6-проверка-согласованности-генерации-подрядчика)
* [7. Визуализация WorkGraph](#7-визуализация-workgraph)
* [8. Вставка графа в граф и реструктуризация](#8-вставка-графа-в-граф-и-реструктуризация)

## 1. Инициализация и базовая генерация

Импорты, генератор, исходный граф, стартовые подрядчики.&#x20;

```python
import random
from itertools import chain
from matplotlib import pyplot as plt

from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.graph import WorkGraph
from sampo.structurator import (
    graph_in_graph_insertion, work_graph_ids_simplification, graph_restructuring
)
from sampo.utilities.visualization.work_graph import work_graph_fig

rand = random.Random(10)
p_rand = SimpleSynthetic(rand=231)
wg = p_rand.work_graph(top_border=3000)
contractors = [p_rand.contractor(i) for i in range(10, 31, 10)]
```

## 2. Базовое планирование HEFT

Расчёт расписания и вывод метрик.&#x20;

```python
schedule = HEFTScheduler().schedule(wg, contractors)[0]

print(len(wg.nodes))
print("\nDefault contractors")
print(f"Execution time: {schedule.execution_time}")
```

## 3. Подрядчик по WorkGraph

Пример оценки с автогенерацией подрядчика под граф. В файле цикл по пустому списку, для эксперимента заполните `pack_counts`.&#x20;

```python
print("\nContractor by work graph")
for pack_counts in [1, 2, 10]:  # в файле: []; замените при необходимости
    contractors = [get_contractor_by_wg(wg, scaler=pack_counts)]
    execution_time = HEFTScheduler().schedule(wg, contractors)[0].execution_time
    print(f"Execution time: {execution_time}, pack count: {pack_counts}")
```

## 4. Расширение номенклатуры работ

Увеличение числа уникальных названий работ функцией `extend_names`.&#x20;

```python
print("\nNames extension")
names_wg = len({n.work_unit.name for n in wg.nodes})
new_wg = extend_names(500, wg, rand)
names_new_wg = len({n.work_unit.name for n in new_wg.nodes})
print(f"works in origin: {names_wg}, pack count: {names_new_wg}")
```

## 5. Расширение номенклатуры ресурсов

Добавление новых типов ресурсов функцией `extend_resources`.&#x20;

```python
print("\nResource extension")
res_names_wg = len({req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])})
new_wg = extend_resources(100, wg, rand)
res_names_new_wg = len({req.kind for req in chain(*[n.work_unit.worker_reqs for n in new_wg.nodes])})
print(f"resources in origin: {res_names_wg}, in new: {res_names_new_wg}")
```

## 6. Проверка согласованности генерации подрядчика

Сопоставление числа типов ресурсов в расширенном графе и у подрядчика, сгенерированного по этому графу.&#x20;

```python
print("\nCheck gen contractor by wg extension")
res_names_new_c = len(set(get_contractor_by_wg(new_wg, scaler=1).workers.keys()))
print(f"works in new by wg: {res_names_new_wg}, by contractor: {res_names_new_c}")
```

## 7. Визуализация WorkGraph

Функция-обёртка для отрисовки графа.&#x20;

```python
def plot_wg(wg: WorkGraph) -> None:
    _ = work_graph_fig(wg, (14, 8), legend_shift=4, show_names=True, text_size=4)
    plt.show()
```

## 8. Вставка графа в граф и реструктуризация

Композиция графов, упрощение идентификаторов, реструктуризация и визуализация этапов.&#x20;

```python
srand = SimpleSynthetic(34)
wg_master = srand.work_graph(cluster_counts=1)
wg_slave  = srand.work_graph(cluster_counts=1)

union_wg = graph_in_graph_insertion(wg_master, wg_master.start, wg_master.finish, wg_slave)
union_wg_simply = work_graph_ids_simplification(union_wg, id_offset=1000)
union_wg_restructured = graph_restructuring(union_wg_simply)

plot_wg(wg_master)
plot_wg(wg_slave)
plot_wg(union_wg)
plot_wg(union_wg_simply)
plot_wg(union_wg_restructured)
```
