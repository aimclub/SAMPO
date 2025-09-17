# Стохастическое ТО самолёта: зоны и GA-планирование

## Оглавление

* [1. Параметры и импорты](#1-параметры-и-импорты)
* [2. Генерация исходного WorkGraph](#2-генерация-исходного-workgraph)
* [3. Дефекты: StructureEstimator](#3-дефекты-structureestimator)
* [4. Зоны и ландшафт](#4-зоны-и-ландшафт)
* [5. Планировщик и оценщик](#5-планировщик-и-оценщик)
* [6. Пайплайн расчёта](#6-пайплайн-расчёта)
* [7. Визуализация Ганта](#7-визуализация-ганта)

---

## 1. Параметры и импорты

```python
from random import Random
import numpy as np

from sampo.generator import SimpleSynthetic, SyntheticGraphType
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import GeneticScheduler
from sampo.schemas import DefaultZoneStatuses, ZoneConfiguration, LandscapeConfiguration
from sampo.schemas.structure_estimator import DefaultStructureGenerationEstimator, DefaultStructureEstimator
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

r_seed = 231
ss = SimpleSynthetic(r_seed)
```



## 2. Генерация исходного WorkGraph

```python
wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200
)
```



## 3. Дефекты: StructureEstimator

* Равномерно добавляются 5 «подработ» на каждый несервисный узел.

```python
rand = Random(r_seed)
generator = DefaultStructureGenerationEstimator(rand)
sub_works = [f"Sub-work {i}" for i in range(5)]

for node in wg.nodes:
    if node.work_unit.is_service_unit:
        continue
    for sub_work in sub_works:
        generator.set_probability(parent=node.work_unit.name,
                                  child=sub_work,
                                  probability=1/len(sub_works))

structure_estimator = DefaultStructureEstimator(generator, rand)
wg_with_defects = structure_estimator.restruct(wg)
contractors = [get_contractor_by_wg(wg_with_defects)]
```



## 4. Зоны и ландшафт

* Одна зона `zone1`, 4 допустимых статуса, единичные переходные стоимости.

```python
class AeroplaneZoneStatuses(DefaultZoneStatuses):
    def statuses_available(self) -> int:
        return 4

zone_names = ['zone1']
zones_count = len(zone_names)

zone_config = ZoneConfiguration(
    start_statuses={zone: 1 for zone in zone_names},
    time_costs=np.array([[1 for _ in range(zones_count)] for _ in range(zones_count)]),
    statuses=AeroplaneZoneStatuses()
)

landscape_config = LandscapeConfiguration(zone_config=zone_config)
```



## 5. Планировщик и оценщик

```python
work_estimator = DefaultWorkEstimator()

genetic_scheduler = GeneticScheduler(
    work_estimator=work_estimator,
    number_of_generation=20,
    mutate_order=0.05,
    mutate_resources=0.005,
    size_of_population=50
)
```



## 6. Пайплайн расчёта

```python
aircraft_project = (
    SchedulingPipeline.create()
    .wg(wg_with_defects)
    .contractors(contractors)
    .work_estimator(work_estimator)
    .landscape(landscape_config)
    .schedule(genetic_scheduler)
    .finish()[0]
)
```



## 7. Визуализация Ганта

```python
merged_schedule = aircraft_project.schedule.merged_stages_datetime_df('2022-01-01')
aircraft_schedule_fig = schedule_gant_chart_fig(
    merged_schedule,
    VisualizationMode.ReturnFig,
    remove_service_tasks=False
)
aircraft_schedule_fig.update_layout(height=1200, width=1600)
aircraft_schedule_fig.show()
```

