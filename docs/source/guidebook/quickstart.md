# Быстрый старт

Покажем установку, подготовку простого примера, запуск планировщика и просмотр результата, а также предоставим
«шпаргалку».

## Установка

SAMPO доступен как пакет Python на PyPI. Установка:

```bash
pip install sampo
```

Убедитесь, что используете Python 3.10 (SAMPO требует 3.10.x).

## Первый план за несколько шагов

Сделаем простейший проект и распишем его:

1. Граф работ — создадим WorkGraph. Для быстрого старта сгенерируем синтетический.
2. Ресурсы — опишем список Contractor с рабочими.
3. Алгоритм — выберем планировщик (эвристика/генетика).
4. Запуск — получим Schedule и посмотрим результат.

1) Генерация WorkGraph. Для простоты воспользуемся генератором

```python
from sampo.generator import SimpleSynthetic, SyntheticGraphType
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

Альтернатива: если у вас уже есть данные, загрузите их из файла:

```python
wg = WorkGraph.load(folder, filename)
```

2) Ресурсы (Contractor’ы)

```python
from sampo.schemas.contractor import Contractor, Worker

# Один подрядчик с 10 работниками общего профиля
contractors = [
    Contractor(
        id="Contractor1",
        workers=[Worker(id="w1", kind="general", count=10)]
    )
]
```

В реальном проекте может быть несколько подрядчиков с разными специализациями. Значение kind должно соответствовать
типам, которых требуют задачи в WorkGraph.

3) Выбор планировщика

```python
from sampo.scheduler.heft import HEFTScheduler

# или: from sampo.scheduler import HEFTScheduler, TopologicalScheduler, GeneticScheduler

scheduler = HEFTScheduler()  # быстрая эвристика для старта
```

Также можно попробовать:

- TopologicalScheduler() — простой базовый порядок
- GeneticScheduler(...) — более глубокий поиск (с настройкой гиперпараметров, например mutate_order, mutate_resources и
  т.п.)

4) Запуск планирования
   Важный момент: метод schedule возвращает список кортежей; нам нужен сам Schedule из первого элемента.

```python
# Планирование: берём первое (лучшее) решение
best_schedule, start_time, timeline, node_order = scheduler.schedule(work_graph, contractors)[0]

print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

Просмотр расписания
У разных версий могут отличаться детали структур расписания. Надёжный способ получить агрегированное представление и
визуализировать — воспользоваться встроенной функцией Ганта:

```python
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

merged = best_schedule.merged_stages_datetime_df(start_date='2025-01-01')
fig = schedule_gant_chart_fig(merged, VisualizationMode.ReturnFig, remove_service_tasks=False)
fig.show()
```

5) (Опционально) Конвейер SchedulingPipeline
   Эквивалент тех же шагов во «флюентном» стиле. finish() возвращает список ScheduledProject, из которого берём [0] и
   затем читаем project.schedule.

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

Подсказки, что попробовать дальше:

- Замените HEFTScheduler на GeneticScheduler для поиска лучших планов.
- Добавьте .optimize_local(...) до/после .schedule(...) для локальных улучшений порядка/расписания.
- Укажите .lag_optimize(...) и .work_estimator(...) при необходимости учитывать дополнительные ограничения и модели
  оценки времени.

## «Шпаргалка»

### Базовые сущности

- WorkGraph — DAG проекта с двумя служебными вершинами start/finish; вершины — GraphNode с вложенным WorkUnit (описание
  работы), рёбра — зависимости с типом EdgeType (не только FS) и поддержкой лагов.
  > WorkGraph — граф работ проекта, где зафиксированы все задачи и их зависимости.

- GraphNode — вершина WorkGraph, хранящая WorkUnit и ссылки на родителей/потомков (с типами рёбер и лагами).
  > GraphNode — контейнер WorkUnit и связей в графе.

- WorkUnit — работа: объём, единицы измерения, приоритет, требования к ресурсам (WorkerReq: вид ресурса и мин/макс
  количество), сервисные флаги и др.
  > WorkUnit — описание одной задачи проекта с её характеристиками и потребностями в ресурсах.

- WorkerReq — требование работы к виду ресурса: specialty/kind, min_count, max_count, объём для нормирования
  длительности.
  > WorkerReq — «каких и сколько ресурсов нужно на эту задачу».

- Contractor — поставщик ресурсов: словари workers: dict[str, Worker] и equipments: dict[str, Equipment]; каждому Worker
  проставляется contractor_id.
  > Contractor — организация или подразделение, предоставляющее людей и технику.

- Worker — ресурс «люди»: имя (специальность), count, contractor_id, распределение производительности (
  Static/Stochastic).
  > Worker — трудовой ресурс с численностью и производительностью (иногда со стоимостной оценкой).

- Equipment — ресурс «техника» с типом и количеством.
  > Equipment — оборудование, используемое в проекте.

---

### Планировщик и результат

- Scheduler — алгоритм планирования. Принимает WorkGraph и список Contractor (а также опционально spec, landscape,
  work_estimator). Возвращает список Schedule. Расширенный метод schedule_with_cache возвращает кортежи (Schedule,
  start_time: Time, Timeline, node_order: list[GraphNode]).
  > Scheduler — модуль, который строит расписание по задачам и ресурсам.
  Примечание: базовый schedule возвращает list[Schedule]; на практике берут первый план: schedule(...)[0].

- Schedule — расписание: обёртка над pandas DataFrame со стартами/финишами, назначениями ресурсов, стоимостью и пр.;
  содержит объекты ScheduledWork. Метрика execution_time — makespan (время завершения последней работы).
  > Schedule — итоговый план выполнения работ с назначенными ресурсами и сроками.

- ScheduledWork — элемент расписания: конкретная работа с назначенными ресурсами, стартом/финишем и длительностью.
  > ScheduledWork — «строка» плана по одной работе.

- ScheduleSpec — спецификация для ограничений/назначений ресурсов на работы (в т.ч. фиксирование объёмов команд).
  > ScheduleSpec — настройки, ограничивающие или уточняющие распределение ресурсов.

- Timeline — внутренняя временная шкала использования ресурсов, которую поддерживает планировщик при вычислениях.
  > Timeline — модель занятости ресурсов во времени.

---

### Оценки и окружение

- WorkTimeEstimator — оценщик длительности работ, подставляется в планировщик через work_estimator.
  > WorkTimeEstimator — функция или модель, определяющая длительность выполнения задачи.

- LandscapeConfiguration / ZoneConfiguration — конфигурация пространственных/зональных ограничений и смен статусов зон (
  при необходимости).
  > Landscape/ZoneConfiguration — описание территории или зоны выполнения работ с ограничениями.

---

### Пайплайн

- Scheduling Pipeline — «флюентный» builder-интерфейс для конфигурирования и запуска планирования:
  SchedulingPipeline.create() → ... → .finish() → ScheduledProject (project.schedule).
  > «Флюентный» builder — цепочка методов, где вы задаёте граф работ, ресурсы и опции, выбираете алгоритм и получаете
  результат как ScheduledProject.

  > Кратко о методах пайплайна (в рекомендуемом порядке):

    1) create() — старт билдера.
    2) name_mapper(fn) — (опционально) нормализация/переименование работ и ресурсов.
    3) wg(x) — задать WorkGraph (объект/таблица/файл).
    4) contractors(x) — задать подрядчиков (список/таблица/генерация).
    5) history(data) — (опционально) исторические данные для калибровки оценок.
    6) work_estimator(est) — (опционально) модель оценки длительностей.
    7) spec(spec) — (опционально) ограничения/фиксация ресурсов и этапов.
    8) landscape(cfg) — (опционально) пространственные/зональные ограничения.
    9) time_shift(offset) — (опционально) сдвиг начала расписания/дат.
    10) lag_optimize(strategy) — (опционально) обработка лагов/разбиение стадий; если не указать — подберётся
        автоматически.
    11) node_order(order|optimizer) — (опционально) зафиксировать/улучшить порядок узлов до планирования.
    12) schedule(scheduler, validate=False) — выбрать алгоритм и построить расписание.
    13) optimize_local(optimizer, …) — (опционально) локальная оптимизация готового плана после schedule.
    14) visualization(opts) — (опционально) визуализация плана.
    15) finish() — получить результат как ScheduledProject (обычно берут [0]).

> Примечание: optimize_local может вызываться также ДО schedule для оптимизации порядка (альтернатива пункту 11).

- ScheduledProject — контейнер с результатом проекта (включая project.schedule), доступен после finish().
  > ScheduledProject — объект с итоговым расписанием проекта.

---

### Многоагентность

- BlockGraph / BlockNode — декомпозиция проекта на блоки (подграфы) с зависимостями между блоками; возможна обратная
  сборка в единый WorkGraph.
  > BlockGraph / BlockNode — представление проекта как набора крупных блоков задач.

- Agent — независимый планировщик с собственными Contractor’ами и Timeline; протокол offer/confirm для выдачи и фиксации
  плана блока.
  > Agent — отдельный планировщик, отвечающий за свой участок проекта.
  Важно: подрядчики разных агентов не должны пересекаться (валидация проверяет уникальность).

- Manager / NeuralManager — координация агентов, распределение блоков и выбор последовательности с учётом зависимостей и
  метрик; NeuralManager использует обучаемую политику выбора.
  > Manager / NeuralManager — управляющий модуль, который согласует работу агентов.
