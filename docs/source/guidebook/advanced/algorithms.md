# Алгоритмы планирования

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