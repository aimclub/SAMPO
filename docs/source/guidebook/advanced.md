# Расширённое использование и настройка

После базового знакомства вы можете тонко настроить SAMPO: выбирать алгоритмы, тюнить параметры, работать с несколькими
критериями и встраивать доменные компоненты.


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
> О структуре CSV:

- Обязательные колонки:
  - activity_id, activity_name, granular_name, volume, measurement, priority
- Зависимости (списки через запятую, длины строго совпадают):
  - predecessor_ids, connection_types, lags
  - Типы связей: FS, SS, FF, IFS, FFS; лаги — числа (обычно 0)
- Опционально (требования по ресурсам в ячейках-словарях):
  - min_req, max_req, req_volume (формат: {"worker_kind": value})
- Примечания:
  - Сервисные узлы start/finish в CSV не нужны — добавляются автоматически
  - В .wg(path, sep=';') укажите разделитель файла; внутри списков — запятая

Мини-пример CSV с двумя задачами и зависимостью B от A (FS, лаг 0):
```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags
A;Task A;A;1.0;unit;0;;;
B;Task B;B;1.0;unit;0;A;FS;0
```
> Пример можно посмотреть в `tests/parser/test_wg.csv`.

### Способ C. Программно из узлов

- Создайте WorkUnit для каждой задачи (с требованиями WorkerReq по необходимости).
- Для зависимостей используйте GraphNode(work_unit, parents), где parents — список кортежей (parent_node, lag, EdgeType).
- Соберите граф: WorkGraph.from_nodes([...]) — start/finish добавятся автоматически.

```python
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time

# Работы (пример: A -> B -> C, FS с лагом 0)
wu_a = WorkUnit(id='A', name='Task A',
                worker_reqs=[WorkerReq(kind='general', volume=Time(10), min_count=2, max_count=4)],
                volume=1.0, is_service_unit=False)

wu_b = WorkUnit(id='B', name='Task B',
                worker_reqs=[], volume=1.0, is_service_unit=False)

wu_c = WorkUnit(id='C', name='Task C',
                worker_reqs=[], volume=1.0, is_service_unit=False)

n_a = GraphNode(wu_a, [])  # корень
n_b = GraphNode(wu_b, [(n_a, 0, EdgeType.FinishStart)])  # A -> B (FS, 0)
n_c = GraphNode(wu_c, [(n_b, 0, EdgeType.FinishStart)])  # B -> C (FS, 0)

wg = WorkGraph.from_nodes([n_a, n_b, n_c])
```

Коротко:
- Типы связей: EdgeType.FinishStart (FS), StartStart (SS), FinishFinish (FF), InseparableFinishStart (IFS), LagFinishStart (FFS).
- Лаг — второй элемент кортежа в parents (число, обычно 0).

## Как создаются/получаются подрядчики (Contractor)

### Вручную:

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

contractor = Contractor(workers={
      'driver': Worker(id='w1', name='driver', count=8, productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)),
      'fitter': Worker(id='w2', name='fitter', count=6, productivity=IntervalGaussian(1.2, 0.1, 0.8, 1.6))},
      id='c1', name='Contractor A')
```

### Генератором по параметру “размер пакета”:
```python
from sampo.generator.base import SimpleSynthetic
ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
```

### По графу работ (из требований):
```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
contractor = get_contractor_by_wg(wg, scaler=1.0, method=ContractorGenerationMethod.AVG)
```

Метод берёт требования по работам (min/max), агрегирует и формирует пул. 

## Как выбрать алгоритм

### Эвристические планировщики (HEFT, HEFTBetween, Topological)
    - Очень быстрые и дают хорошую стартовую базу.
    - HEFT/HEFTBetween ранжируют работы по приоритетам/критичности и оцениваемым временам выполнения.
    - Topological строит порядок по зависимостям без сложной оптимизации.

Импорты:

```python
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.heft import HEFTBetweenScheduler
from sampo.scheduler.topological import TopologicalScheduler
```

### Генетический планировщик
    - Перебирает много альтернатив и часто улучшает результат, но требует больше времени на расчёт.
    - Ключевые параметры:
        - number_of_generation, size_of_population — глубина и ширина поиска;
        - mutate_order — вероятность мутации гена порядка (перестановки работ с сохранением зависимостей). Выше → шире
          поиск по допустимым порядкам, но медленнее сходимость.
        - mutate_resources — вероятность мутации ресурсных генов (перераспределение видов/объемов и подрядчиков в рамках
          min/max и доступности). Выше → больше вариантов параллельности, но при дефиците ресурсов растёт риск
          конфликтов.
        - при необходимости — work_estimator (модель оценки длительностей), seed (для воспроизводимости).

        > * **number_of_generation** — число итераций генетического алгоритма (↑ поколений → лучше результат, но дольше).
        > * **size_of_population** — размер популяции (↑ особей → выше разнообразие, но дороже по времени/памяти).
        > * **mutate_order** — вероятность мутации порядка работ (↑ → активнее поиск, но медленнее сходимость).
        > * **mutate_resources** — вероятность мутации распределения ресурсов (↑ → больше параллельности, но риск конфликтов).

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

### Многоагентное планирование 
  - делит граф на блоки, применяет разные стратегии и объединяет результат. Полезно для очень
  больших проектов и гибридных стратегий; требует большего сетапа (модуль `sampo.scheduler.multi_agency`).

#### Пример без разбиения на блоки: 
- два агента с разными планировщиками «соревнуются» за лучший план (аукцион).

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

# 2) Универсальный подрядчик с запасом по всем требуемым видам работников
kinds = {req.kind for node in wg.nodes for req in node.work_unit.worker_reqs}
cid = str(uuid4())
workers = {k: Worker(str(uuid4()), k, 50, contractor_id=cid) for k in kinds}
contractors = [Contractor(id=cid, name="Universal", workers=workers, equipments={})]

# 3) Два агента с разными стратегиями
agents = [
    Agent("HEFT", HEFTScheduler(), contractors),
    Agent("Topological", TopologicalScheduler(), contractors),
]
manager = StochasticManager(agents)

# 4) Аукцион: кто даст расписание с меньшим окончанием — тот победил
start, end, schedule, winner = manager.run_auction(wg)
print("Победил агент:", winner.name, "Мейкспан:", end - start)
```

#### Пример многоагентного планирования с разбиением на блоки. 

```python
from random import Random
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.scheduler.multi_agency.block_generator import generate_blocks, SyntheticBlockGraphType

# 1) Генерируем граф блоков (BlockGraph).
#    Каждый блок — это отдельный WorkGraph, который можно планировать изолированно.
#    Межблочные зависимости задаются через edge_prob.
seed = 231
rand = Random(seed)
bg = generate_blocks(
    SyntheticBlockGraphType.RANDOM,
    n_blocks=4,                       # сколько блоков
    type_prop=[1, 1, 1],              # пропорции типов внутренних графов (General/Parallel/Sequential)
    count_supplier=lambda i: (10, 15),# размер каждого блока (нижняя/верхняя границы числа работ)
    edge_prob=0.3,                    # вероятность ребра между блоками (межблочные зависимости)
    rand=rand
)

# 2) Создаём простых подрядчиков для агентов
ss = SimpleSynthetic(rand)
contractor_a = ss.contractor(40)
contractor_b = ss.contractor(40)

# 3) Два агента с разными стратегиями планирования и своими ресурсами.
agents = [
    Agent("HEFT", HEFTScheduler(), [contractor_a]),
    Agent("Topo", TopologicalScheduler(), [contractor_b]),
]
manager = StochasticManager(agents)

# 4) Менеджер идёт по блокам в топологическом порядке и проводит «аукцион» для каждого блока:
#    - Вычисляет parent_time = max(окончаний всех родительских блоков) + 1.
#    - Каждый агент предлагает расписание (offer) c учётом parent_time и своей текущей занятости.
#    - Выбирается предложение с минимальным end_time и подтверждается (confirm).
scheduled_blocks = manager.manage_blocks(bg)

# 5) Результаты: кто какой блок выиграл и глобальные времена.
print("Scheduled blocks:")
for block_id, sblock in scheduled_blocks.items():
    print(f"Block {block_id}: agent={sblock.agent.name}, "
          f"start={sblock.start_time}, end={sblock.end_time}, "
          f"duration={sblock.duration}")

# Суммарная длительность проекта.
makespan = max(sb.end_time for sb in scheduled_blocks.values())
print("Project makespan:", makespan)
```
> Коротко про блоки и менеджера:

>  - Блок — это самостоятельный подграф работ (WorkGraph), который планируется целиком как единица.
>  - BlockGraph — это DAG из блоков; ребро A → B означает: B можно начать только после завершения A.
>  - Менеджер идёт по блокам в топологическом порядке.
>  - Для каждого блока считает parent_time = максимум окончаний его родительских блоков.
>  - Агенты делают офферы: строят расписание блока с учётом parent_time и своей текущей занятости.
>  - Побеждает агент с минимальным временем окончания блока; его план подтверждается, таймлайн обновляется.
>  - Зачем: масштабируемость, параллельность, изолируемость пересчётов и удобство управления этапами/зонами.
---
