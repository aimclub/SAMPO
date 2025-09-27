# WorkGraph

## Термины

**WorkGraph** — ориентированный ациклический граф (DAG) проекта с двумя служебными вершинами `start` / `finish`; при
сборке из набора узлов сервисные вершины добавляются автоматически.

> Фиксирует задачи и их зависимости (узлы — `GraphNode`, рёбра — связи с типом и лагом). Циклы запрещены.


**GraphNode** — вершина графа, содержащая `WorkUnit` и ссылки на родительские/дочерние узлы (каждая связь имеет тип и
лаг).

> Контейнер задачи и её входящих/исходящих зависимостей.

**WorkUnit** — задача проекта с объёмом и набором требований к ресурсам.

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

> mode: 'general' or 'sequence' or 'parallel - the type of the returned graph
> cluster_counts: Number of clusters for the graph
> bottom_border: bottom border for number of works for the graph
> top_border: top border for number of works for the graph

---

### Способ B. Загрузить из CSV

```python
from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.heft import HEFTScheduler

project = (SchedulingPipeline.create()
.wg(wg='tests/parser/test_wg.csv', sep=';', all_connections=True)
.lag_optimize(LagOptimizationStrategy.TRUE)
.schedule(HEFTScheduler())
.finish()[0])
```

> all_connections: Пытаться достроить или дополнить связи между работами, если их не хватает во входных данных.
> sep: Разделитель столбцов CSV
> LagOptimizationStrategy: Shows should graph be lag-optimized or not.
If not defined, pipeline should search the best variant of this argument in result.

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