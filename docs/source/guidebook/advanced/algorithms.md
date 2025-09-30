# Алгоритмы планирования

## Как выбрать алгоритм

### Эвристические планировщики (HEFT, HEFTBetween, Topological)

- Быстро дают рабочий план.
- HEFT/HEFTBetween рассчитывают порядок задач и примерное время выполнения, чтобы закончить быстрее.
- Topological просто выстраивает задачи по зависимостям, без сложной оптимизации.

Импорты:

```python
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.heft import HEFTBetweenScheduler
from sampo.scheduler.topological import TopologicalScheduler
```

---

### Генетический планировщик

- Пробует много разных вариантов и часто находит план с меньшим временем завершения проекта. Работает дольше простых.
- Главные настройки:
    - `number_of_generation` — сколько раз улучшать решения (больше — выше шанс улучшить, но дольше).
    - `size_of_population` — сколько вариантов держать одновременно (больше — больше идей, но медленнее и больше
      памяти).
    - `mutate_order` — как часто менять порядок задач (сохраняя зависимости). Больше — шире поиск, но может сходиться
      медленнее.
    - `mutate_resources` — как часто менять распределение ресурсов/подрядчиков. Больше — выше шанс параллелить, но при
      дефиците ресурсов может чаще «конфликтовать».
    - Дополнительно: `work_estimator`, `seed` (для воспроизводимости).

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

Разбивает проект на части (блоки), планирует их разными способами и собирает общий план. Полезно на больших проектах и
при комбинировании стратегий (`sampo.scheduler.multi_agency`).

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

# 2) Подрядчик с нужными типами работников
kinds = {req.kind for node in wg.nodes for req in node.work_unit.worker_reqs}
cid = str(uuid4())
workers = {k: Worker(str(uuid4()), k, 50, contractor_id=cid) for k in kinds}
contractors = [Contractor(id=cid, name="Universal", workers=workers, equipments={})]

# 3) Два агента с разными подходами
agents = [
    Agent("HEFT", HEFTScheduler(), contractors),
    Agent("Topological", TopologicalScheduler(), contractors),
]
manager = StochasticManager(agents)

# 4) «Аукцион»: агенты предлагают свои планы, выбираем лучший по времени завершения
start, end, schedule, winner = manager.run_auction(wg)
print("Победил агент:", winner.name, "Время завершения:", end - start)
```

### С разбиением на блоки

```python
from random import Random
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.scheduler.multi_agency.block_generator import generate_blocks, SyntheticBlockGraphType

# 1) Строим «граф блоков» (каждый блок — это свой маленький WorkGraph)
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

# 3) Агенты с разными алгоритмами
agents = [
    Agent("HEFT", HEFTScheduler(), [contractor_a]),
    Agent("Topo", TopologicalScheduler(), [contractor_b]),
]
manager = StochasticManager(agents)

# 4) Планируем блоки по порядку, учитывая зависимости между ними
scheduled_blocks = manager.manage_blocks(bg)

# 5) Итоги: кто какой блок взял и когда он выполнялся
print("Scheduled blocks:")
for block_id, sblock in scheduled_blocks.items():
    print(
        f"Block {block_id}: agent={sblock.agent.name}, start={sblock.start_time}, end={sblock.end_time}, duration={sblock.duration}")

project_finish = max(sb.end_time for sb in scheduled_blocks.values())
print("Время завершения проекта:", project_finish)
```

Коротко:

- Блок — самостоятельная часть проекта (свой небольшой граф работ).
- Граф блоков — связки между блоками: «A → B» значит, что блок B можно начинать после завершения A.
- Менеджер ждёт, пока закончатся все предыдущие блоки, собирает предложения от агентов и выбирает то, где блок
  завершится раньше.
