# Быстрый старт

Покажем установку, подготовку простого примера, запуск планировщика и просмотр результата.

## Установка

SAMPO доступен в виде пакета Python (требуется Python 3.10.x):

```bash
pip install sampo
```

## Первый план за несколько шагов

Сделаем простейший проект и распишем его:

1) Граф работ — создадим WorkGraph. Для быстрого старта сгенерируем синтетический.
2) Ресурсы — опишем список Contractor с рабочими.
3) Алгоритм — выберем планировщик (эвристика/генетика).
4) Запуск — получим Schedule и посмотрим результат.

---

### 1) Генерация WorkGraph (быстрый способ — генератор)

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType
from sampo.schemas.graph import WorkGraph

# Инициализация синтетического генератора
synthetic = SimpleSynthetic()

# Сгенерируем небольшой граф: ~2 кластера по 5–8 задач
work_graph: WorkGraph = synthetic.work_graph(
    mode=SyntheticGraphType.GENERAL,  # тип структуры: GENERAL / PARALLEL / SEQUENTIAL
    cluster_counts=2,  # 2 кластера
    bottom_border=5,  # в кластере 5–8 задач
    top_border=8
)
print(f"Generated a WorkGraph with {len(work_graph.nodes)} tasks.")
```

---

### 2) Ресурсы (Contractor’ы)

Важно: синтетический граф использует типовые профессии `driver`, `fitter`, `manager`, `handyman`, `electrician`,
`engineer`.  
Подрядчик должен содержать работников для каждого требуемого вида, а словарь `workers` индексируется по имени вида
ресурса (`req.kind`).

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

# Зададим по нескольку работников каждого нужного типа
workers = [
    Worker(id="w_driver", name="driver", count=20),
    Worker(id="w_fitter", name="fitter", count=20),
    Worker(id="w_manager", name="manager", count=10),
    Worker(id="w_handyman", name="handyman", count=20),
    Worker(id="w_electrician", name="electrician", count=10),
    Worker(id="w_engineer", name="engineer", count=10),
]

# Один подрядчик с полным пулом работников
contractors = [
    Contractor(
        id="c1",
        name="General Contractor",
        # Ключи — имена видов ресурсов (совпадают с WorkerReq.kind)
        workers={w.name: w for w in workers}
    )
]
```

Альтернатива: автогенерация подрядчика «по графу», чтобы ресурсы точно покрывали требования:

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractors = [get_contractor_by_wg(
    work_graph,
    scaler=1.0,  # множитель мощностей (>= 1.0)
    method=ContractorGenerationMethod.AVG,  # усреднение между min/max потребностями
    contractor_id="c1",
    contractor_name="General Contractor"
)]
```

---

### 3) Выбор планировщика

```python
from sampo.scheduler.heft import HEFTScheduler

# также доступны:
# from sampo.scheduler.topological import TopologicalScheduler
# from sampo.scheduler.genetic import GeneticScheduler

scheduler = HEFTScheduler()  # быстрая эвристика для старта
```

---

### 4) Запуск планирования

Метод `schedule(...)` возвращает список объектов `Schedule`. Берём первое (лучшее) решение:

```python
best_schedule = scheduler.schedule(work_graph, contractors)[0]
print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

Если нужна дополнительная информация (время, таймлайн, порядок узлов), используйте расширенный метод:

```python
best_schedule, finish_time, timeline, node_order = scheduler.schedule_with_cache(work_graph, contractors)[0]
print(f"Makespan: {best_schedule.execution_time}")
```

---

### Просмотр расписания (диаграмма Ганта)

Надёжный способ — получить агрегированное представление и визуализировать его:

```python
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

merged = best_schedule.merged_stages_datetime_df(offset='2025-01-01')

fig = schedule_gant_chart_fig(
    merged,
    visualization=VisualizationMode.ReturnFig,
    color_type='contractor'  # можно сменить раскраску при необходимости
)
fig.show()
```

- Если хотите увидеть только свои задачи без «внутренних» технических, возьмите таблицу best_schedule.pure_schedule_df.
- Для диаграммы Ганта обычно используют календарное представление best_schedule.merged_stages_datetime_df целиком.

---

### 5) (Опционально) Конвейер SchedulingPipeline

Эквивалент тех же шагов во «флюентном» стиле. `finish()` возвращает список `ScheduledProject`, из которого берём `[0]` и
читаем `project.schedule`.

```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft import HEFTScheduler

project = (SchedulingPipeline.create()
.wg(work_graph)
.contractors(contractors)
.schedule(HEFTScheduler())
.finish()[0])

print(f"Project duration: {project.schedule.execution_time}")
```