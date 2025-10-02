# WorkGraph

## Термины

- **WorkGraph** — «карта проекта» в виде ациклического графа. Две служебные точки — начало и конец — добавляются
  автоматически.
- **GraphNode** — узел графа: содержит задачу (`WorkUnit`) и ссылки на связи с родителями/детьми (у связи есть тип и
  пауза).
- **WorkUnit** — описание самой работы: объём и требования к ресурсам (какие люди/сколько).

---

## Как собрать WorkGraph

### А) Сгенерировать автоматически (быстрый старт)

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType

ss = SimpleSynthetic()
wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,  # тип структуры: General / Parallel / Sequential
    cluster_counts=10,  # число кластеров (групп работ)
    bottom_border=100,  # нижняя граница количества работ
    top_border=200  # верхняя граница количества работ
)
```

* `mode` — форма графа (общий, параллельный, последовательный).
* `cluster_counts` — сколько групп работ в графе.
* `bottom_border` / `top_border` — диапазон количества задач.

---

### Б) Загрузить из CSV

```python
from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.heft import HEFTScheduler

project = (
    SchedulingPipeline.create()
    .wg(wg='tests/parser/test_wg.csv', sep=';', all_connections=True)
    .lag_optimize(LagOptimizationStrategy.TRUE)  # AUTO / NONE / TRUE
    .schedule(HEFTScheduler())
    .finish()[0]
)

wg = project.wg  # готовый WorkGraph
schedule = project.schedule
```

* `all_connections=True` — попытаться «достроить» связи, если во входных данных чего‑то не хватает.
* `sep=';'` — разделитель колонок в CSV.
* `LagOptimizationStrategy` — как обращаться с технологическими паузами (можно доверить выбор «AUTO»).

> Если используете автогенерацию подрядчика через `get_contractor_by_wg`, **ресурсные колонки можно не
добавлять**. Достаточно обязательных полей и зависимостей.

---

## Структура CSV

### Обязательные колонки

| Поле            | Описание                                                             |
|-----------------|----------------------------------------------------------------------|
| `activity_id`   | Уникальный идентификатор задачи                                      |
| `activity_name` | Человекочитаемое название задачи                                     |
| `granular_name` | Короткий код/метка задачи                                            |
| `volume`        | Объём работы (число)                                                 |
| `measurement`   | Единица измерения (например `unit`, `m3`, `pcs`)                     |
| `priority`      | Целое число для приоритизации (если не нужно — ставьте `0` для всех) |

### Зависимости

Три синхронных списка (длины совпадают):

* `predecessor_ids` — ID предшественников через запятую
* `connection_types` — типы связей (FS, SS, FF и т.п.)
* `lags` — паузы (числа)

> Пустая ячейка = просто **три разделителя подряд без пробелов** (`;;;`).
>
> Пример:
> `predecessor_ids="A,B"`, `connection_types="FS,SS"`, `lags="0,3"`

### Ресурсы (опционально)

| Поле         | Формат                                        |
|--------------|-----------------------------------------------|
| `min_req`    | JSON-словарь минимально требуемых ресурсов    |
| `max_req`    | JSON-словарь максимально допустимых ресурсов  |
| `req_volume` | JSON-словарь норм/объёмов для расчёта времени |

* Используйте **валидный JSON** с двойными кавычками:
  `{"welder": 2, "driver": 1}`
* Если этих колонок нет, планировщик использует автогенерацию ресурсов (например, через `get_contractor_by_wg`).
* Ключи должны совпадать с `Worker.name`.

---

### Минимальный CSV (без ресурсов)

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags
A;Task A;A;1.0;unit;0;;;
B;Task B;B;1.0;unit;0;A;FS;0
C;Task C;C;1.0;unit;0;B;FS;0
```

### CSV с ресурсами (пример)

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags;min_req;max_req;req_volume
A;Site prep;A;1.0;unit;0;;;;"{""driver"":1,""handyman"":1}";"{""driver"":2,""handyman"":2}";"{""driver"":8,""handyman"":8}"
B;Foundation;B;1.0;unit;0;A;FS;0;"{""fitter"":2,""engineer"":1}";"{""fitter"":4,""engineer"":2}";"{""fitter"":12,""engineer"":12}"
```

---

### Автогенерация подрядчика для CSV-графа

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
from sampo.scheduler.heft import HEFTScheduler

wg = project.wg

contractors = [get_contractor_by_wg(
    wg,
    scaler=1.0,
    method=ContractorGenerationMethod.AVG,
    contractor_id="c_csv",
    contractor_name="CSV Contractor"
)]

scheduler = HEFTScheduler()
best_schedule, *_ = scheduler.schedule_with_cache(wg, contractors)[0]
print(f"Makespan: {best_schedule.execution_time}")
```

---

### В) Собрать программно из узлов

```python
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time

# Пример цепочки: A -> B -> C (FS, паузы 0)
wu_a = WorkUnit(
    id='A', name='Task A',
    worker_reqs=[WorkerReq(kind='driver', volume=Time(10), min_count=2, max_count=4)],
    volume=1.0, is_service_unit=False
)
wu_b = WorkUnit(id='B', name='Task B', worker_reqs=[], volume=1.0, is_service_unit=False)
wu_c = WorkUnit(id='C', name='Task C', worker_reqs=[], volume=1.0, is_service_unit=False)

n_a = GraphNode(wu_a, [])
n_b = GraphNode(wu_b, [(n_a, 0, EdgeType.FinishStart)])  # FS
n_c = GraphNode(wu_c, [(n_b, 0, EdgeType.FinishStart)])  # FS

wg = WorkGraph.from_nodes([n_a, n_b, n_c])
```

* Обязательные поля `WorkUnit`: `id`, `name`.
* Для экспорта полезно также указывать `volume`, `measurement`, `priority`.
