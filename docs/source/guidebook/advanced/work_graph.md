# WorkGraph

## Термины

- WorkGraph — это «карта проекта» в виде графа без циклов. Есть две служебные точки: начало и конец (добавляются сами).
- GraphNode — узел графа: внутри лежит задача (`WorkUnit`) и ссылки на связи с родителями/детьми (у связи есть тип и
  пауза).
- WorkUnit — описание самой работы: объём и требования к ресурсам (какие люди/сколько и т.п.).

---

## Как собрать WorkGraph

### А) Сгенерировать автоматически (быстрый старт)

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType

ss = SimpleSynthetic()
wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,  # тип структуры: General / Parallel / Sequential
    cluster_counts=10,  # сколько «кластеров» (логических групп работ)
    bottom_border=100,  # нижняя граница количества работ
    top_border=200  # верхняя граница количества работ
)
```

Коротко про параметры:

- `mode` — форма графа (общий, параллельный, последовательный).
- `cluster_counts` — сколько групп работ в графе.
- `bottom_border` / `top_border` — примерно сколько работ генерировать.

---

### Б) Загрузить из CSV

```python
from sampo.pipeline import SchedulingPipeline
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.heft import HEFTScheduler

project = (SchedulingPipeline.create()
.wg(wg='tests/parser/test_wg.csv', sep=';', all_connections=True)
.lag_optimize(LagOptimizationStrategy.TRUE)  # можно AUTO/NONE/TRUE
.schedule(HEFTScheduler())
.finish()[0])

wg = project.wg  # готовый WorkGraph
schedule = project.schedule
```

Пояснения:

- `all_connections=True` — попытаться «достроить» связи, если во входных данных чего‑то не хватает.
- `sep=';'` — разделитель колонок в CSV.
- `LagOptimizationStrategy` — как обращаться с технологическими паузами (можно доверить выбор «AUTO»).

## Структура CSV

- Обязательные колонки: `activity_id, activity_name, granular_name, volume, measurement, priority`
    - `activity_id` — уникальный идентификатор задачи (строка/число). Должен быть уникальным во всём файле.
    - `activity_name` — понятное человеку название задачи (строка).
    - `granular_name` — короткий код/метка задачи (строка). Если не используете — можно дублировать `activity_id`/
      `activity_name`.
    - `volume` — объём работы (число). Используется оценщиком времени; смысл объёма зависит от вашей предметной области.
    - `measurement` — единица измерения объёма (строка), например: `unit`, `m3`, `m`, `pcs`.
    - `priority` — целое число для приоритизации. Если нет особого порядка — ставьте `0` для всех.

- Зависимости (списки через запятую, длины совпадают): `predecessor_ids, connection_types, lags`
    - Это три «синхронных» списка. Элементы с одинаковым индексом относятся к одной и той же связи.
        - Пример:  
          `predecessor_ids="A,B"`, `connection_types="FS,SS"`, `lags="0,3"`  
          означает две связи: A —(FS, 0)→ текущая задача и B —(SS, 3)→ текущая задача.
    - Пустая ячейка в `predecessor_ids` означает «нет предшественников».
    - Допустимые типы связей перечислены ниже. Паузы (`lags`) — числа (обычно `0`).
    - Важно: длина всех трёх списков должна совпадать, порядок элементов — согласованный.

- Про типы связи можно прочитать в отдельной главе.

- По ресурсам (опционально, словари в ячейках): `min_req`, `max_req`, `req_volume` вида `{"worker_kind": value}`
    - Формат — JSON‑подобный словарь в строке. Ключ — имя профессии/ресурса, значение — число.
        - `min_req` — минимально требуемое количество данного ресурса (например, людей нужной профессии).
        - `max_req` — максимально допустимое количество (верхняя граница бригады).
        - `req_volume` — «норма/объём» по данному виду ресурса (как правило, влияет на расчёт времени через оценщик).
    - Примеры значений ячейки:
        - `{"welder": 2, "driver": 1}`
        - `{"operator": 3}`
    - Если колонки отсутствуют или ячейка пуста — считается, что ограничений по этому виду нет (или берутся значения по
      умолчанию).
    - Имена ключей должны совпадать с `Worker.name` у ваших подрядчиков. Иначе планировщик не сопоставит требуемые и
      доступные ресурсы.

Мини‑пример зависимостей (B после A по FS, пауза 0):

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags
A;Task A;A;1.0;unit;0;;;
B;Task B;B;1.0;unit;0;A;FS;0
```

---

### В) Собрать программно из узлов

- Сначала создайте `WorkUnit` для каждой работы (при необходимости укажите `WorkerReq`).
- Связи задаются при создании `GraphNode`: список родителей в виде кортежей `(parent_node, lag, EdgeType)`.
- Сборка: `WorkGraph.from_nodes([...])` — служебные `start/finish` добавятся сами.

```python
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time

# A -> B -> C (FS, паузы 0)
wu_a = WorkUnit(
    id='A', name='Task A',
    worker_reqs=[WorkerReq(kind='general', volume=Time(10), min_count=2, max_count=4)],
    volume=1.0, is_service_unit=False
)
wu_b = WorkUnit(id='B', name='Task B', worker_reqs=[], volume=1.0, is_service_unit=False)
wu_c = WorkUnit(id='C', name='Task C', worker_reqs=[], volume=1.0, is_service_unit=False)

n_a = GraphNode(wu_a, [])
n_b = GraphNode(wu_b, [(n_a, 0, EdgeType.FinishStart)])  # FS
n_c = GraphNode(wu_c, [(n_b, 0, EdgeType.FinishStart)])  # FS

wg = WorkGraph.from_nodes([n_a, n_b, n_c])
```

- Обязательные поля у `WorkUnit`: `id`, `name`. Для экспорта удобно также задавать `volume`, `measurement`, `priority`.

---