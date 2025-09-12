# Планирование синтетического графа (simple\_synthetic\_graph\_scheduling.py)

## Оглавление

* [1. Параметры и импорты](#1-параметры-и-импорты)
* [2. Генерация синтетического WorkGraph](#2-генерация-синтетического-workgraph)
* [3. Диагностика атрибутов графа](#3-диагностика-атрибутов-графа)
* [4. Подрядчик по графу](#4-подрядчик-по-графу)
* [5. Планирование](#5-планирование)
* [6. Визуализация диаграммы Ганта](#6-визуализация-диаграммы-ганта)
* [7. Валидация результата](#7-валидация-результата)

---

## 1. Параметры и импорты

* Планировщик: `HEFTScheduler`. Дата старта: `"2023-01-01"`. Визуализация: `VisualizationMode.ShowFig` или `SaveFig` с именем файла.&#x20;

```python
from itertools import chain
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.time import Time

scheduler = HEFTScheduler()
start_date = "2023-01-01"
visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = './output/synth_schedule_gant_chart.png'
```

## 2. Генерация синтетического WorkGraph

* Пределы генерации: `works_count_top_border=2000`, `uniq_works=300`, `uniq_resources=100`. Фиксированный `rand=31`.&#x20;

```python
synth_works_top_border = 2000
synth_unique_works = 300
synth_resources = 100

srand = SimpleSynthetic(rand=31)
wg = srand.advanced_work_graph(
    works_count_top_border=synth_works_top_border,
    uniq_works=synth_unique_works,
    uniq_resources=synth_resources
)
```

## 3. Диагностика атрибутов графа

* Подсчёт работ, уникальных названий и типов ресурсов. Жёсткие проверки на верхние границы.&#x20;

```python
works_count = len(wg.nodes)
work_names_count = len(set(n.work_unit.name for n in wg.nodes))
res_kind_count = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
print(works_count, work_names_count, res_kind_count)

assert (works_count     <= synth_works_top_border * 1.1)
assert (work_names_count <= synth_unique_works)
assert (res_kind_count   <= synth_resources)
```

## 4. Подрядчик по графу

* Автогенерация подрядчика, покрывающего ресурсы графа: `get_contractor_by_wg(wg)`.&#x20;

```python
contractors = [get_contractor_by_wg(wg)]
```

## 5. Планирование

* Расчёт расписания и преобразование к датам.&#x20;

```python
schedule = scheduler.schedule(wg, contractors)[0]
schedule_df = schedule.merged_stages_datetime_df(start_date)
print(schedule.execution_time)
```

## 6. Визуализация диаграммы Ганта

* Построение диаграммы. Можно показывать или сохранять. Сервисные задачи удаляются.&#x20;

```python
gant_fig = schedule_gant_chart_fig(
    schedule_df,
    fig_file_name=gant_chart_filename,
    visualization=visualization_mode,
    remove_service_tasks=True
)
```

## 7. Валидация результата

* Проверка, что планирование завершилось успешно: время не бесконечность.&#x20;

```python
assert schedule.execution_time != Time.inf(), \
    f'Scheduling failed on {scheduler.scheduler_type.name}'
```
