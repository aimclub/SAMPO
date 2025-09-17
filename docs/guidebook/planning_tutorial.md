# Туториал по планированию в SAMPO

Краткий практический гид по подготовке данных, инициализации планировщиков и построению расписания

## Оглавление

* [1. Подготовка графа работ и подрядчиков](#1-подготовка-графа-работ-и-подрядчиков)

  * [1.1 Загрузка из CSV](#11-загрузка-из-csv)
  * [1.2 Загрузка из сериализованного JSON](#12-загрузка-из-сериализованного-json)

    * [1.2.1 WorkGraph → DataFrame](#121-workgraph--dataframe)
    * [1.2.2 Автоподбор подрядчиков по WorkGraph](#122-автоподбор-подрядчиков-по-workgraph)
    * [1.2.3 Загрузка расписания из JSON](#123-загрузка-расписания-из-json)
  * [1.x Интеграция в STAIRS](#1x-интеграция-в-stairs)
* [2. Инициализация алгоритмов планирования](#2-инициализация-алгоритмов-планирования)

  * [2.1 Пользовательский WorkTimeEstimator](#21-пользовательский-worktimeestimator)
  * [2.2 Эвристические и топологические планировщики](#22-эвристические-и-топологические-планировщики)
  * [2.3 Генетический алгоритм: гиперпараметры](#23-генетический-алгоритм-гиперпараметры)
* [3. Построение расписания через пайплайн](#3-построение-расписания-через-пайплайн)

  * [3.1 Настройка пайплайна](#31-настройка-пайплайна)

    * [3.1.1 Восстановление структуры и лагов по истории](#311-восстановление-структуры-и-лагов-по-истории)
    * [3.1.2 Передача подрядчиков](#312-передача-подрядчиков)
    * [3.1.3 Передача исторических данных](#313-передача-исторических-данных)
    * [3.1.4 Реструктуризация графа](#314-реструктуризация-графа)
    * [3.1.5 Подключение оценщика времени](#315-подключение-оценщика-времени)
    * [3.1.6 Построение расписания](#316-построение-расписания)
  * [3.2 Полное планирование и экспорт](#32-полное-планирование-и-экспорт)

---

## Импорт и базовая настройка

```python
# Планировщики
from sampo.scheduler.genetic.base import HEFTScheduler, HEFTBetweenScheduler, GeneticScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.topological.base import RandomizedTopologicalScheduler

# Схемы и пайплайн
from sampo.schemas import ScheduledProject
from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy

# Подрядчики и утилиты
from sampo.generator.environment import ContractorGenerationMethod, get_contractor_by_wg
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

# Пользовательский оценщик времени/ресурсов (пример из ноутбука)
from field_dev_resources_time_estimator import FieldDevWorkEstimator

# Табличные данные
import pandas as pd
import numpy as np

# Пути данных
DATA_PATH = 'data/12/field_development/'
DUMPS_PATH = DATA_PATH + 'dumps/'
CSV_PATH = ''
```

[к оглавлению](#оглавление)

---

## 1. Подготовка графа работ и подрядчиков

### 1.1 Загрузка из CSV

Колонки по работам и связям. Для связей используются `predecessor_ids`, тип соединения и лаг. Если связи будут восстанавливаться по истории, допустимо отсутствие колонок связей.

```python
# Пример из ноутбука
df = pd.read_csv(CSV_PATH + 'electroline_field_dev_demo.csv', sep=';')
df.head()
```

[к оглавлению](#оглавление)

---

### 1.2 Загрузка из сериализованного JSON

Импорт проекта целиком удобен для повторной визуализации и сравнения: структура работ (`WorkGraph`) и, при наличии, расписание (`Schedule`).

#### 1.2.1 WorkGraph → DataFrame

```python
# Допустим, у нас есть объект ScheduledProject (загружен из JSON или получен после планирования)
# Извлекаем WorkGraph и переводим его в DataFrame для анализа/восстановления лагов
df_with_req = scheduling_project.wg.to_frame(save_req=True)
df_with_req.head()
```

[к оглавлению](#оглавление)

---

#### 1.2.2 Автоподбор подрядчиков по WorkGraph

```python
# project_wg — объект WorkGraph
from sampo.generator.environment import get_contractor_by_wg, ContractorGenerationMethod

# Даже одного подрядчика оборачиваем списком
project_contractors = [get_contractor_by_wg(
    wg=project_wg,
    method=ContractorGenerationMethod.AVG,
    contractor_name='Main contractor'
)]

# Короткий вариант
project_contractors = [get_contractor_by_wg(wg=project_wg, contractor_name='Main contractor')]
```

[к оглавлению](#оглавление)

---

#### 1.2.3 Загрузка расписания из JSON

```python
# Если из сериализованного проекта уже получен Schedule:
raw_schedule = scheduling_project.schedule

# Переводим в человекочитаемые даты, задав дату старта проекта
project_schedule = raw_schedule.merged_stages_datetime_df('2022-09-01')
project_schedule.head()

# Визуализация диаграммы Ганта средствами SAMPO
_ = schedule_gant_chart_fig(
    schedule_dataframe=project_schedule,
    visualization=VisualizationMode.ShowFig,  # также доступны ReturnFig и SaveFig
    remove_service_tasks=False
)
```

[к оглавлению](#оглавление)

---

### 1.x Интеграция в STAIRS

```python
# Подготовка DataFrame к загрузке в STAIRS
df = pd.read_csv(CSV_PATH + 'electroline_field_dev_demo.csv', sep=';')
df['measurement'] = df['granular_measurement']
df['activity_id'] = list(range(200, 200 + len(df)))  # пример автонумерации

# Подключение к БД для пользовательского оценщика
db_url = "postgresql+psycopg2://testuser:pwd@10.32.15.30:25432/test"
project_work_estimator = FieldDevWorkEstimator(url=db_url)
```

После `.finish()` пайплайна доступен `ScheduledProject`. Из него: `Schedule` и `wg.to_frame(save_req=True)` для СППР.

[к оглавлению](#оглавление)

---

## 2. Инициализация алгоритмов планирования

### 2.1 Пользовательский WorkTimeEstimator

Один экземпляр оценщика используется всеми планировщиками.

```python
# Пример из ноутбука
project_work_estimator = FieldDevWorkEstimator()  # модель, реализующая интерфейс WorkTimeEstimator
```

[к оглавлению](#оглавление)

---

### 2.2 Эвристические и топологические планировщики

```python
# Топологические
topo_scheduler = TopologicalScheduler(work_estimator=project_work_estimator)
rand_topo_scheduler = RandomizedTopologicalScheduler(work_estimator=project_work_estimator)

# HEFT-эвристики
heft_scheduler = HEFTScheduler(work_estimator=project_work_estimator)
heft_between_scheduler = HEFTBetweenScheduler(work_estimator=project_work_estimator)
```

[к оглавлению](#оглавление)

---

### 2.3 Генетический алгоритм: гиперпараметры

Два режима из ноутбука: с авто-выбором числа поколений и с явными параметрами.

```python
# Вариант с авто-подбором количества поколений (пример)
custom_number_of_generation = np.random.randint(10, 25)
custom_mutate_order = 0.05
custom_mutate_resources = 0.005
custom_size_of_population = 50

# Инициализация генетического планировщика
genetic_scheduler = GeneticScheduler(
    number_of_generation=custom_number_of_generation,
    mutate_order=custom_mutate_order,
    mutate_resources=custom_mutate_resources,
    size_of_population=custom_size_of_population,
    work_estimator=project_work_estimator
)
```

```python
# Вариант с фиксированными настройками (пример)
custom_number_of_generation = 20
custom_mutate_order = 0.1       # из диапазона ~0.05..0.15
custom_mutate_resources = 0.01  # из диапазона ~0.005..0.015
custom_size_of_population = 50

genetic_scheduler = GeneticScheduler(
    number_of_generation=custom_number_of_generation,
    mutate_order=custom_mutate_order,
    mutate_resources=custom_mutate_resources,
    size_of_population=custom_size_of_population,
    work_estimator=project_work_estimator
)
```

[к оглавлению](#оглавление)

---

## 3. Построение расписания через пайплайн

### 3.1 Настройка пайплайна

Последовательная передача графа, подрядчиков, истории и оценщика. Опционально — оптимизация лагов.

```python
# Старт пайплайна
scheduling_pipeline = SchedulingPipeline.create()
```

[к оглавлению](#оглавление)

---

#### 3.1.1 Восстановление структуры и лагов по истории

```python
# df_with_req — DataFrame графа (см. 1.2.1)
# Параметры восстановления: пример с отключением автосмены типов связей
scheduling_pipeline = scheduling_pipeline.wg(
    wg=df_with_req,
    all_connections=False,
    # change_connections_* — используйте режимы восстановления только при неполных связях
)
```

[к оглавлению](#оглавление)

---

#### 3.1.2 Передача подрядчиков

```python
# Передаем список подрядчиков (см. 1.2.2)
scheduling_pipeline = scheduling_pipeline.contractors(project_contractors)
```

[к оглавлению](#оглавление)

---

#### 3.1.3 Передача исторических данных

```python
# Исторические проекты для восстановления типов связей и лагов
history_df = pd.read_csv('historical_projects_data.csv', sep=';')
scheduling_pipeline = scheduling_pipeline.history(history_df, sep=';')
```

[к оглавлению](#оглавление)

---

#### 3.1.4 Реструктуризация графа

Оптимизация параллелизма с учетом связей F–F(S) и временных лагов.

```python
# Явно укажите стратегию. В ноутбуке рекомендуется задавать явно.
scheduling_pipeline = scheduling_pipeline.lag_optimize(LagOptimizationStrategy.TRUE)
```

[к оглавлению](#оглавление)

---

#### 3.1.5 Подключение оценщика времени

```python
# Единый оценщик на всех этапах
scheduling_pipeline = scheduling_pipeline.work_estimator(project_work_estimator)
```

[к оглавлению](#оглавление)

---

#### 3.1.6 Построение расписания

```python
# Выбор планировщика: topo_scheduler | rand_topo_scheduler | heft_scheduler | heft_between_scheduler | genetic_scheduler
schedule_project = scheduling_pipeline.schedule(genetic_scheduler).finish()[0]

# Быстрый просмотр
schedule_project.schedule.pure_schedule_df.head()

# Перевод к датам и визуализация
project_schedule = schedule_project.schedule.merged_stages_datetime_df('2022-09-01')
_ = schedule_gant_chart_fig(project_schedule, VisualizationMode.ShowFig, remove_service_tasks=False)
```

[к оглавлению](#оглавление)

---

### 3.2 Полное планирование и экспорт

```python
# Сквозной прогон
schedule_project2 = SchedulingPipeline.create() \
    .wg(df_with_req, all_connections=False) \
    .contractors(project_contractors) \
    .history(history_df, sep=';') \
    .lag_optimize(LagOptimizationStrategy.TRUE) \
    .work_estimator(project_work_estimator) \
    .schedule(genetic_scheduler) \
    .finish()[0]

# Итоговое расписание
raw_project_schedule = schedule_project2.schedule
project_schedule = raw_project_schedule.merged_stages_datetime_df('2022-09-01')

# Экспорт структуры проекта для СППР
schedule_project2.dump('.', 'gas_network_full_connections_genetic_upd')
```

[к оглавлению](#оглавление)
