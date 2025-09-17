# Конфигурация ландшафта и планирование в SAMPO (LandscapeConfiguration)

## Оглавление

* [1. Данные и ландшафт](#1-данные-и-ландшафт)

  * [1.1 Генерация графа и материалов](#11-генерация-графа-и-материалов)
  * [1.2 Построение ландшафта](#12-построение-ландшафта)
* [2. Планировщик](#2-планировщик)
* [3. Подрядчик](#3-подрядчик)
* [4. Пайплайн и визуализация](#4-пайплайн-и-визуализация)
* [5. Диагностика](#5-диагностика)
* [6. Полный пример](#6-полный-пример)

---

## 1. Данные и ландшафт

### 1.1 Генерация графа и материалов

* Синтетика: `SimpleSynthetic(rand=31) → small_work_graph()`.&#x20;
* Материалы на узлы: `set_materials_for_wg(wg)`.&#x20;

```python
from sampo.generator import SimpleSynthetic

ss = SimpleSynthetic(rand=31)
wg = ss.small_work_graph()
wg = ss.set_materials_for_wg(wg)
```

### 1.2 Построение ландшафта

* Конфигурация площадок: `synthetic_landscape(wg)`.&#x20;

```python
landscape = ss.synthetic_landscape(wg)
```

---

## 2. Планировщик

* Генетический алгоритм с ограниченными гиперпараметрами.
* Предупреждение: большие значения поколений и популяции замедляют расчёт при сложном ландшафте.&#x20;

```python
from sampo.scheduler import GeneticScheduler

scheduler = GeneticScheduler(
    number_of_generation=1,
    mutate_order=0.05,
    mutate_resources=0.005,
    size_of_population=10,
)
```

---

## 3. Подрядчик

* Автоподбор по требованиям `WorkGraph`: `get_contractor_by_wg(wg)`.&#x20;

```python
from sampo.generator.environment import get_contractor_by_wg

contractors = [get_contractor_by_wg(wg)]
```

---

## 4. Пайплайн и визуализация

* Используется `DefaultInputPipeline`.
* Визуализация диаграммы Ганта; режимы: `VisualizationMode.ShowFig | SaveFig`.
* Параметры: дата старта, имя файла при `SaveFig`.&#x20;

```python
from sampo.pipeline.default import DefaultInputPipeline
from sampo.utilities.visualization import VisualizationMode

start_date = "2023-01-01"
visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = "./output/synth_schedule_gant_chart.png"

project = (
    DefaultInputPipeline()
    .wg(wg)
    .contractors(contractors)
    .landscape(landscape)
    .schedule(scheduler)
    .visualization(start_date)  # возвращает список проектов
)[0].show_gant_chart()
```

---

## 5. Диагностика

* Число платформ и наличие материалов на каждом узле.&#x20;

```python
platform_number = len(landscape.platforms)
is_all_nodes_have_materials = all(node.work_unit.need_materials() for node in wg.nodes)
print(f"LandscapeConfiguration: {platform_number} platforms, "
      f"All nodes have materials: {is_all_nodes_have_materials}")
```

---

## 6. Полный пример

```python
from sampo.generator import SimpleSynthetic
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline.default import DefaultInputPipeline
from sampo.scheduler import GeneticScheduler
from sampo.utilities.visualization import VisualizationMode

start_date = "2023-01-01"
visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = "./output/synth_schedule_gant_chart.png"

ss = SimpleSynthetic(rand=31)
wg = ss.small_work_graph()
wg = ss.set_materials_for_wg(wg)
landscape = ss.synthetic_landscape(wg)

scheduler = GeneticScheduler(
    number_of_generation=1,
    mutate_order=0.05,
    mutate_resources=0.005,
    size_of_population=10,
)

platform_number = len(landscape.platforms)
is_all_nodes_have_materials = all(node.work_unit.need_materials() for node in wg.nodes)
print(f"LandscapeConfiguration: {platform_number} platforms, "
      f"All nodes have materials: {is_all_nodes_have_materials}")

contractors = [get_contractor_by_wg(wg)]

project = (
    DefaultInputPipeline()
    .wg(wg)
    .contractors(contractors)
    .landscape(landscape)
    .schedule(scheduler)
    .visualization(start_date)
)[0].show_gant_chart()
```
