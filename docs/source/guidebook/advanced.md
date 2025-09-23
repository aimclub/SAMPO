# Расширённое использование и настройка

Здесь объясняются ключевые концепции, показаны примеры кода и даны рекомендации по расширенному использованию
библиотеки.

---

## Базовые сущности

* **WorkGraph** — ориентированный ациклический граф работ проекта (DAG) с двумя служебными вершинами `start`/`finish` (
  всегда содержит служебные узлы; при сборке из узлов они добавляются автоматически). Вершины — `GraphNode`, внутри —
  `WorkUnit`, рёбра — зависимости (поддерживаются типы связей и лаги).

  > WorkGraph фиксирует все задачи и зависимости проекта.

* **Тип связи** — вид предшествования между работами в WorkGraph:

    * **FS (Finish–Start)**: `S(B) ≥ F(A) + лаг`.
    * **SS (Start–Start)**: `S(B) ≥ S(A) + лаг`.
    * **FF (Finish–Finish)**: `F(B) ≥ F(A) + лаг`.
    * **IFS (Inseparable Finish–Start)**: неразрывная FS — B сразу после A без разрывов; узлы образуют слитную цепочку.
    * **FFS (LagFinishStart)**: поточная связь; потомок стартует после выполнения предком части объёма, равной лагу.

  > Примечание: в проверках «жестких зависимостей» обычно учитываются FS/IFS/FFS; SS/FF могут обрабатываться отдельно в
  планировщиках.

* **Лаг** — сдвиг ограничения:

    * Для FS/SS/FF — числовой сдвиг во времени (обычно `0`). Знак: `>0` — задержка, `<0` — «лид».
    * Для FFS — лаг в единицах объёма предка (например, км). Движок делит работу на стадии; старт потомка — после стадии
      объёмом `лаг`. При отключённой оптимизации лагов FFS ведёт себя как обычный FS.

* **GraphNode** — вершина `WorkGraph`, хранящая `WorkUnit` и ссылки на родителей/потомков (с типом связи и лагом).

  > Контейнер `WorkUnit` и его связей.

* **WorkUnit** — описание работы: `id`, `name`, `volume`, `measurement` (единицы), `priority`, требования к ресурсам (
  `WorkerReq`), сервисные флаги.

  > «Что за задача и что ей нужно».

* **WorkerReq** — требование к ресурсу: `kind`, `min_count`, `max_count`, `volume` (норма для расчёта длительности).

  > «Каких специалистов и сколько нужно».

* **Contractor** — поставщик ресурсов: `workers: dict[str, Worker]`, `equipments: dict[str, Equipment]`. При
  инициализации `contractor_id` у `Worker` проставляется автоматически.

  > Организация/подразделение, предоставляющее людей и технику.

* **Worker** — трудовой ресурс: `id`, `name` (специализация), `count`, `contractor_id`, `productivity` (распределение),
  опционально `cost_one_unit`.

  > `Worker.name` должен совпадать с `WorkerReq.kind` для корректного сопоставления.

* **Equipment** — технический ресурс: тип и количество.

  > Оборудование, используемое в проекте.

* **Scheduler** — алгоритм планирования. Принимает `WorkGraph` и `list[Contractor]` (опц. `spec`, `landscape`,
  `work_estimator`). Базовые реализации/методы могут возвращать `Schedule` и доп. структуры (например,
  `schedule_with_cache` → список кортежей `(Schedule, Time, Timeline, ...)`). В практическом использовании берут
  результат/первый план.

  > Планировщик строит расписание.

* **Schedule / ScheduledWork / Timeline** — итоговый план (старты/финиши, назначения ресурсов, внутренняя шкала
  занятости).

  > `Schedule.execution_time` — мейкспан (длительность проекта).

> Примечание: **DAG** — ориентированный ацикличный граф; циклы запрещены, что гарантирует упорядочиваемость плана.

---

## Как собрать WorkGraph

### Способ A. Сгенерировать синтетический граф

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType

ss = SimpleSynthetic()
wg = ss.work_graph(mode=SyntheticGraphType.GENERAL,
                   cluster_counts=10,
                   bottom_border=100,
                   top_border=200)
```

Коротко: удобно для тестов и прототипов; управляем размером и «кластерностью».

---

### Способ B. Загрузить из CSV

```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft import HEFTScheduler
from sampo.pipeline.lag_optimization import LagOptimizationStrategy

project = (SchedulingPipeline.create()
.wg(wg='tests/parser/test_wg.csv', sep=';', all_connections=True)
.lag_optimize(LagOptimizationStrategy.TRUE)
.schedule(HEFTScheduler())
.finish()[0])
```

**Структура CSV**

* Обязательные колонки: `activity_id, activity_name, granular_name, volume, measurement, priority`.
* Зависимости (списки через запятую, длины совпадают): `predecessor_ids, connection_types, lags`.
* Типы: `FS`, `SS`, `FF`, `IFS`, `FFS`; лаги — числа (обычно `0`).
* Опционально (требования в словарях-ячейках): `min_req`, `max_req`, `req_volume` вида `{"worker_kind": value}`.
* Примечания:

    * `start/finish` в CSV не нужны — добавляются автоматически.
    * В `.wg(path, sep=';')` `sep` — разделитель файла; внутри списков — запятая.
    * Ошибка `Parameter 'id' unfilled/name unfilled` часто означает пустые `activity_id`/`activity_name`.

**Мини-пример (B зависит от A по FS, лаг 0)**

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags
A;Task A;A;1.0;unit;0;;;
B;Task B;B;1.0;unit;0;A;FS;0
```

> Полный пример: `tests/parser/test_wg.csv`.

---

### Способ C. Программно из узлов

* Создайте `WorkUnit` для каждой задачи (с `WorkerReq` при необходимости).
* Зависимости: `GraphNode(work_unit, parents)`, где `parents` — список кортежей `(parent_node, lag, EdgeType)`.
* Сборка: `WorkGraph.from_nodes([...])` — `start/finish` добавятся автоматически.

```python
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time

# A -> B -> C (FS, лаг 0)
wu_a = WorkUnit(id='A', name='Task A',
                worker_reqs=[WorkerReq(kind='general', volume=Time(10), min_count=2, max_count=4)],
                volume=1.0, is_service_unit=False)
wu_b = WorkUnit(id='B', name='Task B', worker_reqs=[], volume=1.0, is_service_unit=False)
wu_c = WorkUnit(id='C', name='Task C', worker_reqs=[], volume=1.0, is_service_unit=False)

n_a = GraphNode(wu_a, [])
n_b = GraphNode(wu_b, [(n_a, 0, EdgeType.FinishStart)])  # FS
n_c = GraphNode(wu_c, [(n_b, 0, EdgeType.FinishStart)])  # FS

wg = WorkGraph.from_nodes([n_a, n_b, n_c])
```

Коротко по зависимостям:

* Типы: `FinishStart (FS)`, `StartStart (SS)`, `FinishFinish (FF)`, `InseparableFinishStart (IFS)`,
  `LagFinishStart (FFS)`.
* Лаг — второй элемент кортежа в `parents`.
* Обязательные поля `WorkUnit`: `id`, `name`. Для экспорта также задавайте `volume`, `measurement`, `priority`.

---

## Подрядчики (Contractor)

### Вручную

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

contractor = Contractor(
    workers={
        'driver': Worker(id='w1', name='driver', count=8, productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)),
        'fitter': Worker(id='w2', name='fitter', count=6, productivity=IntervalGaussian(1.2, 0.1, 0.8, 1.6)),
    },
    id='c1',
    name='Contractor A'
)
```

> Важно:
>
> * Ключи словаря `workers` должны совпадать с `Worker.name`.
> * `WorkerReq.kind` должен совпадать с `Worker.name`, иначе бригада не подберётся.
> * `contractor_id` у `Worker` берётся из `Contractor.id`.
> * При необходимости задайте `cost_one_unit` явно.

### Генератором по параметру «размер пакета»

```python
from sampo.generator.base import SimpleSynthetic

ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
```

### По графу работ (из требований)

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractor = get_contractor_by_wg(wg, scaler=1.0, method=ContractorGenerationMethod.AVG)
```

Коротко: агрегирует `min/max` из `WorkerReq` по задачам и формирует пул ресурсов.

---

## Выбор алгоритма

### Эвристические планировщики (HEFT, HEFTBetween, Topological)

* Быстрые стартовые решения.
* HEFT/HEFTBetween ранжируют узлы по приоритетам и оценкам длительностей.
* Topological строит порядок по зависимостям без сложной оптимизации.

Импорты:

```python
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.heft import HEFTBetweenScheduler
from sampo.scheduler.topological import TopologicalScheduler
```

---

### Генетический планировщик

* Перебирает множество альтернатив и часто улучшает мейкспан. Требует больше времени.
* Ключевые параметры:

    * `number_of_generation` — число итераций (↑ поколений → выше шанс улучшить, но дольше расчёт).
    * `size_of_population` — размер популяции (↑ особей → выше диверсификация, но дороже по времени/памяти).
    * `mutate_order` — вероятность мутации порядка работ при сохранении зависимостей (↑ → шире поиск, медленнее
      сходимость).
    * `mutate_resources` — вероятность мутации распределения ресурсов/подрядчиков (↑ → больше параллельности; при
      дефиците растёт риск конфликтов).
    * Дополнительно: `work_estimator`, `seed`.

Пример:

```python
from sampo.scheduler.genetic import GeneticScheduler

scheduler = GeneticScheduler(
    number_of_generation=50,
    size_of_population=100,
    mutate_order=0.1,
    mutate_resources=0.1
)
```

---

## Многоагентное планирование

Делит граф на блоки, применяет разные стратегии и объединяет результат. Полезно для крупных проектов и гибридных
стратегий (`sampo.scheduler.multi_agency`).

### «Аукцион» без разбиения на блоки

```python
from uuid import uuid4
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

# 1) Небольшой граф работ
ss = SimpleSynthetic(231)
wg = ss.work_graph(bottom_border=30, top_border=40)

# 2) Универсальный подрядчик по требуемым видам работников
kinds = {req.kind for node in wg.nodes for req in node.work_unit.worker_reqs}
cid = str(uuid4())
workers = {k: Worker(str(uuid4()), k, 50, contractor_id=cid) for k in kinds}
contractors = [Contractor(id=cid, name="Universal", workers=workers, equipments={})]

# 3) Два агента
agents = [
    Agent("HEFT", HEFTScheduler(), contractors),
    Agent("Topological", TopologicalScheduler(), contractors),
]
manager = StochasticManager(agents)

# 4) Аукцион
start, end, schedule, winner = manager.run_auction(wg)
print("Победил агент:", winner.name, "Мейкспан:", end - start)
```

### С разбиением на блоки

```python
from random import Random
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.scheduler.multi_agency.block_generator import generate_blocks, SyntheticBlockGraphType

# 1) Граф блоков (BlockGraph)
seed = 231
rand = Random(seed)
bg = generate_blocks(
    SyntheticBlockGraphType.RANDOM,
    n_blocks=4,
    type_prop=[1, 1, 1],
    count_supplier=lambda i: (10, 15),
    edge_prob=0.3,
    rand=rand
)

# 2) Подрядчики для агентов
ss = SimpleSynthetic(rand)
contractor_a = ss.contractor(40)
contractor_b = ss.contractor(40)

# 3) Агенты
agents = [
    Agent("HEFT", HEFTScheduler(), [contractor_a]),
    Agent("Topo", TopologicalScheduler(), [contractor_b]),
]
manager = StochasticManager(agents)

# 4) Планирование по блокам в топологическом порядке
scheduled_blocks = manager.manage_blocks(bg)

# 5) Итоги
print("Scheduled blocks:")
for block_id, sblock in scheduled_blocks.items():
    print(
        f"Block {block_id}: agent={sblock.agent.name}, start={sblock.start_time}, end={sblock.end_time}, duration={sblock.duration}")

makespan = max(sb.end_time for sb in scheduled_blocks.values())
print("Project makespan:", makespan)
```

Коротко:

* **Блок** — самостоятельный подграф (`WorkGraph`) как единица планирования.
* **BlockGraph** — DAG блоков; `A → B` означает старт `B` после завершения `A`.
* Менеджер считает `parent_time = max(окончаний родителей)`; агенты делают офферы; побеждает минимальный `end_time`.

---

## Пайплайн: рекомендуемый порядок шагов

1. `create()` — старт билдера.
2. `name_mapper(mapper|path)` — (опц.) нормализация имён работ и ресурсов.
3. `wg(x, all_connections=False, change_connections_info=False, sep=',')` — задать `WorkGraph` (объект/таблица/файл).
4. `contractors(x)` — задать/сгенерировать подрядчиков.
5. `history(data, sep=',')` — (опц.) связи из истории, если в графе их нет.
6. `work_estimator(est)` — (опц.) модель длительностей.
7. `spec(spec)` — (опц.) ограничения/фиксации ресурсов и этапов.
8. `landscape(cfg)` — (опц.) пространственные/зональные ограничения.
9. `time_shift(offset: Time)` — (опц.) сдвиг начала.
10. `lag_optimize(strategy)` — стратегия оптимизации лагов (поддерживаются `TRUE`, `FALSE`, `AUTO`/`NONE`).
11. `node_order(orders: list[list[GraphNode]])` — (опц.) зафиксировать порядок узлов.
12. `optimize_local(optimizer: OrderLocalOptimizer, area: range)` — (опц.) локальная оптимизация порядка до `schedule`.
13. `schedule(scheduler, validate=False)` — построить расписание.
14. `optimize_local(optimizer: ScheduleLocalOptimizer, area: range)` — (опц.) локальная оптимизация после `schedule`.
15. `visualization(start_date)` — (опц.) визуализация:
    * `.shape(w, h)`, `.color_type('contractor' | ... )`
    * `.show_gant_chart()`, `.show_resource_charts()`, `.show_work_graph()`
16. `finish()` — `list[ScheduledProject]` (обычно `[0]`).

> `ScheduledProject` содержит итоговый `project.schedule` с `execution_time` и деталями по работам.

---

## Оценка длительностей и производительности

* **WorkTimeEstimator** считает длительность задачи и передаётся через `work_estimator`.
* **Режимы производительности** работников:
    * `Static` — детерминированный (среднее), для воспроизводимых запусков.
    * `Stochastic` — случайный (выборка из распределений), для анализа рисков.
* Интуитивная формула: `duration ≈ volume / (sum(productivity × count) × communication_coeff)`.
    * `productivity` — из распределений у `Worker` (например, `IntervalGaussian`).
    * `communication_coeff` снижает эффективность при росте команды к `max_count` (`WorkerReq`).
* Воспроизводимость: используйте `Static` или задайте `sigma=0` у распределений.
* Практика:
    * Реалистично задавайте `min_count/max_count` в `WorkerReq`.
    * В `Stochastic` усредняйте по нескольким прогонам.

Пример подключения:

```python
from sampo.scheduler.heft import HEFTScheduler
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.schemas.resources import WorkerProductivityMode

est = DefaultWorkEstimator()
est.set_productivity_mode(WorkerProductivityMode.Static)

scheduler = HEFTScheduler(work_estimator=est)
```