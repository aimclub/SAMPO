# Быстрый старт

Покажем установку, подготовку простого примера, запуск планировщика и просмотр результата.

## Установка

SAMPO доступен в виде пакета для языка Python. Установка:

```bash
pip install sampo
```

SAMPO требует версию Python 3.10.x

## Первый план за несколько шагов

Сделаем простейший проект и распишем его:

1. Граф работ — создадим WorkGraph. Для быстрого старта сгенерируем синтетический.
2. Ресурсы — опишем список Contractor с рабочими.
3. Алгоритм — выберем планировщик (эвристика/генетика).
4. Запуск — получим Schedule и посмотрим результат.

1) Генерация WorkGraph. Для простоты воспользуемся генератором

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType
from sampo.schemas.graph import WorkGraph

# Инициализация синтетического генератора
synthetic = SimpleSynthetic()

# Сгенерируем небольшой граф: ~2 кластера по 5–8 задач
work_graph: WorkGraph = synthetic.work_graph(
    mode=SyntheticGraphType.GENERAL,  # тип структуры
    cluster_counts=2,  # 2 кластера
    bottom_border=5,  # в кластере 5–8 задач
    top_border=8
)
print(f"Generated a WorkGraph with {len(work_graph.nodes)} tasks.")
```

2) Ресурсы (Contractor’ы)

Важно: синтетический граф использует типовые профессии `driver`, `fitter`, `manager`, `handyman`, `electrician`,
`engineer`. Подрядчик должен содержать работников для каждого вида требования, а словарь `workers` должен быть
индексирован по имени вида ресурса (т.е. `req.kind`).

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

Альтернатива: можно автоматически сгенерировать подрядчика «по графу», чтобы ресурсы точно покрывали требования:

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractors = [get_contractor_by_wg(
    work_graph,
    scaler=1.0,  # множитель мощностей (>= 1.0)
    method=ContractorGenerationMethod.AVG,  # брать среднее между min/max потребностями
    contractor_id="c1",
    contractor_name="General Contractor"
)]
```

3) Выбор планировщика

```python
from sampo.scheduler.heft import HEFTScheduler

# или: from sampo.scheduler import HEFTScheduler, TopologicalScheduler, GeneticScheduler

scheduler = HEFTScheduler()  # быстрая эвристика для старта
```

4) Запуск планирования  
   Метод `schedule` возвращает список решений; берём первое.

```python
# Планирование: берём первое (лучшее) решение
best_schedule, start_time, timeline, node_order = scheduler.schedule(work_graph, contractors)[0]

print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

Просмотр расписания  
Надёжный способ получить агрегированное представление и визуализировать — воспользоваться встроенной функцией Ганта:

```python
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

merged = best_schedule.merged_stages_datetime_df(start_date='2025-01-01')
fig = schedule_gant_chart_fig(merged, VisualizationMode.ReturnFig, remove_service_tasks=False)
fig.show()
```

5) (Опционально) Конвейер SchedulingPipeline  
   Эквивалент тех же шагов во «флюентном» стиле. `finish()` возвращает список `ScheduledProject`, из которого берём
   `[0]` и затем читаем `project.schedule`.

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