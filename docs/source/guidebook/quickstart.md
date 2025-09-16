## Быстрый старт

Покажем установку, подготовку простого примера, запуск планировщика и просмотр результата. Целевая аудитория — те, кому нужен быстрый результат с минимумом настроек.

### Базовая терминология

* **WorkGraph** — граф проекта (DAG): вершины — **работы** (задачи), рёбра — **предшествование**. Узлы содержат требования к ресурсам, объёмы и пр.
* **Contractor** — поставщик ресурсов (бригада, подрядчик). Список Contractor’ов описывает весь доступный пул.
* **Worker/Resource** — тип ресурса и его мощность внутри Contractor’а (напр., 100 «строителей»). Требования задач должны совпадать с имеющимися типами.
* **Scheduler** — реализация алгоритма, принимающая WorkGraph и Contractor’ов и возвращающая Schedule (HEFTScheduler, GeneticScheduler и др.).
* **Schedule** — результат планирования: порядок, времена старта/финиша, назначенные ресурсы; общая **execution\_time** и иные метрики. Часто `schedule(...)` возвращает список (напр., Парето-решения) — обычно берут первый `[0]` как основной.
* **Scheduling Pipeline** — «флюентный» интерфейс пошагового построения и запуска планирования.

Далее — практическое знакомство.

### Установка

SAMPO доступен как пакет Python на PyPI. Установка:

```bash
pip install sampo
```

Убедитесь, что используете Python 3.10 (SAMPO требует 3.10.x).

### Первый план за несколько шагов

Сделаем простейший проект и распишем его:

1. **Граф работ** — создадим `WorkGraph`. Для быстрого старта сгенерируем синтетический.
2. **Ресурсы** — опишем список `Contractor` с рабочими.
3. **Алгоритм** — выберем планировщик (эвристика/генетика).
4. **Запуск** — получим Schedule и посмотрим результат.

**1. Генерация WorkGraph.** Для простоты воспользуемся генератором:

```python
from sampo.generator import SimpleSynthetic
from sampo.schemas.graph import WorkGraph

# Инициализация синтетического генератора
synthetic = SimpleSynthetic()

# Сгенерируем небольшой граф: ~10 задач в 2 кластерах
work_graph: WorkGraph = synthetic.work_graph(
    mode="General",       # общий тип структуры
    cluster_counts=2,     # 2 кластера задач
    bottom_border=5,      # в каждом кластере 5–8 задач
    top_border=8
)
print(f"Generated a WorkGraph with {len(work_graph.nodes)} tasks.")
```

`SimpleSynthetic().work_graph(...)` создаёт случайный WorkGraph. Мы задали два кластера (условно две фазы) и диапазон задач в кластере. Требования к ресурсам заполняются генератором.

*(Альтернатива: если у вас уже есть данные, можно загрузить их через `WorkGraph.load(folder, filename)`. Для быстрого старта синтетика удобнее.)*

**2. Ресурсы (Contractor’ы).** Опишем доступные ресурсы:

```python
from sampo.schemas.contractor import Contractor, Worker

# Один подрядчик с 10 работниками общего профиля
contractors = [
    Contractor(
        id="Contractor1",
        workers=[
            Worker(id="w1", kind="general", count=10)
        ]
    )
]
```

В реальном проекте может быть несколько подрядчиков с разными специализациями. Поле `kind` должно соответствовать типам, которые требуют задачи.

**3. Выбор планировщика.** Для старта — эвристика **HEFTScheduler**:

```python
from sampo.scheduler.heft import HEFTScheduler

# Инициализация HEFT без доп. параметров
scheduler = HEFTScheduler()
```

Можно попробовать и `TopologicalScheduler()` (простой базовый порядок) или `GeneticScheduler()` (более глубокий поиск). Для начала HEFT даёт быстрый и приличный результат. (В продвинутом разделе — настройка параметров, напр., `GeneticScheduler(population_size=50, mutate_order=0.1, ...)`.)

**4. Запуск планирования.**

```python
# Запуск
schedules = scheduler.schedule(work_graph, contractors)
best_schedule = schedules[0]   # берём первое (лучшее) решение

print(f"Total tasks scheduled: {len(best_schedule.works)}")
print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

`schedule` возвращает список расписаний. Часто это один элемент, поэтому берём `[0]`. Выводим число задач и суммарную длительность проекта.

Далее можно просмотреть задачи/назначения:

```python
for task in best_schedule.works:
    print(f"Task {task.id} starts at t={task.start_time}, ends at t={task.finish_time}, "
          f"assigned to resource type '{task.work_unit.worker_reqs[0].kind}'")
```

*(Названия атрибутов могут отличаться; ориентируйтесь на API SAMPO. Пример предполагает наличие `start_time`, `finish_time` и `work_unit` с требованиями к ресурсам.)*

**5. (Опционально) Конвейер SchedulingPipeline:**

```python
from sampo.pipeline import SchedulingPipeline

schedule = SchedulingPipeline.create() \
    .wg(work_graph) \
    .contractors(contractors) \
    .schedule(HEFTScheduler()) \
    .finish()[0]

print(f"Project duration: {schedule.execution_time}")
```

Это эквивалент предыдущих шагов, но в «флюентном» стиле.
