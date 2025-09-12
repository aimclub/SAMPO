# Визуализация в SAMPO

## Оглавление

* [1. Планирование (Scheduling)](#1-планирование-scheduling)

  * [1.1 Генерация данных](#11-генерация-данных)
  * [1.2 Расчёт расписания](#12-расчёт-расписания)
* [2. Визуализация WorkGraph](#2-визуализация-workgraph)
* [3. Диаграмма Ганта проекта](#3-диаграмма-ганта-проекта)
* [4. Занятость ресурсов](#4-занятость-ресурсов)

  * [4.1 По работам](#41-по-работам)
  * [4.2 По датам](#42-по-датам)

---

## 1. Планирование (Scheduling)

### 1.1 Генерация данных

```python
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
from sampo.generator import SyntheticGraphType

r_seed = 231
ss = SimpleSynthetic(r_seed)

simple_wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)

contractors = [get_contractor_by_wg(simple_wg, method=ContractorGenerationMethod.AVG)]
```

### 1.2 Расчёт расписания

```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler

scheduler = HEFTScheduler()

project = (
    SchedulingPipeline.create()
    .wg(simple_wg)
    .contractors(contractors)
    .lag_optimize(LagOptimizationStrategy.TRUE)
    .schedule(scheduler)
    .finish()[0]
)

schedule = project.schedule
```

---

## 2. Визуализация WorkGraph

```python
from sampo.utilities.visualization.work_graph import work_graph_fig

fig = work_graph_fig(simple_wg, (10, 10))
fig.show()
```

---

## 3. Диаграмма Ганта проекта

```python
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

# переводим расписание в реальные даты
merged_schedule = schedule.merged_stages_datetime_df('2022-01-01')

fig = schedule_gant_chart_fig(
    schedule_dataframe=merged_schedule,
    visualization=VisualizationMode.ShowFig,
    remove_service_tasks=False,
)
```

---

## 4. Занятость ресурсов

### 4.1 По работам

```python
from sampo.utilities.visualization.resources import resource_employment_fig, EmploymentFigType
from sampo.utilities.visualization.base import VisualizationMode

fig = resource_employment_fig(
    schedule=merged_schedule,
    fig_type=EmploymentFigType.WorkLabeled,
    vis_mode=VisualizationMode.ShowFig,
)
```

### 4.2 По датам

```python
from sampo.utilities.visualization.resources import resource_employment_fig, EmploymentFigType
from sampo.utilities.visualization.base import VisualizationMode

fig = resource_employment_fig(
    schedule=merged_schedule,
    fig_type=EmploymentFigType.DateLabeled,
    vis_mode=VisualizationMode.ShowFig,
)
```
