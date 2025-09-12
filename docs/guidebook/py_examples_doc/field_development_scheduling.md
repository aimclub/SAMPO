# Планирование работ по разработке месторождения (field\_development\_scheduling.py)

## Оглавление

* [1. Параметры и импорты](#1-параметры-и-импорты)
* [2. Загрузка WorkGraph](#2-загрузка-workgraph)
* [3. Подрядчики: файл или автогенерация](#3-подрядчики-файл-или-автогенерация)
* [4. Реструктуризация графа](#4-реструктуризация-графа)
* [5. Визуализация структуры](#5-визуализация-структуры)
* [6. Настройки планировщика и вывода](#6-настройки-планировщика-и-вывода)
* [7. Расчёт расписания и даты](#7-расчёт-расписания-и-даты)
* [8. Диаграмма Ганта](#8-диаграмма-ганта)

---

## 1. Параметры и импорты

Подавление предупреждений `matplotlib`, импорты планировщика, визуализации и структуризации.&#x20;

```python
import warnings
from matplotlib import pyplot as plt

from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.structurator.base import graph_restructuring
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.utilities.visualization.work_graph import work_graph_fig

warnings.filterwarnings("ignore")
```

## 2. Загрузка WorkGraph

Чтение структуры задач из подготовленного JSON (`./field_development_tasks_structure.json`).&#x20;

```python
field_development_wg = WorkGraph.load("./", "field_development_tasks_structure")
```

## 3. Подрядчики: файл или автогенерация

Флаг выбора источника. По умолчанию автогенерация под граф со скейлером ресурсов `3`.&#x20;

```python
use_contractors_from_file = False

if use_contractors_from_file:
    contractors = Contractor.load("./", "field_development_contractors_info")
else:
    contractors = [get_contractor_by_wg(field_development_wg, scaler=3)]
```

## 4. Реструктуризация графа

Оптимизация структуры графа с учётом лагов.&#x20;

```python
structured_wg = graph_restructuring(field_development_wg, use_lag_edge_optimization=True)
```

## 5. Визуализация структуры

Отрисовка `WorkGraph`: размер 20×10, подписи узлов, сдвиг легенды.&#x20;

```python
_ = work_graph_fig(structured_wg, (20, 10), legend_shift=4, show_names=True, text_size=6)
plt.show()
```

## 6. Настройки планировщика и вывода

Выбор алгоритма, режим визуализации и дата старта; флаги сериализации подготовлены, но в скрипте не используются.&#x20;

```python
scheduler_type = HEFTScheduler()
graph_structure_optimization = True
csv_required = False
json_required = True

visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = "./output/schedule_gant_chart.png"
start_date = "2023-01-01"
```

## 7. Расчёт расписания и даты

Планирование с валидацией и преобразование расписания к календарным датам.&#x20;

```python
schedule = scheduler_type.schedule(structured_wg, contractors, validate=True)[0]
schedule_df = schedule.merged_stages_datetime_df(start_date)
```

## 8. Диаграмма Ганта

Построение диаграммы с опцией удаления сервисных задач; показ или сохранение файла по режиму.&#x20;

```python
gant_fig = schedule_gant_chart_fig(
    schedule_df,
    fig_file_name=gant_chart_filename,
    visualization=visualization_mode,
    remove_service_tasks=True,
)
```
