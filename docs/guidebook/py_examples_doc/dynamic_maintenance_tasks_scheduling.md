# Динамическое планирование регламентных задач (multi-agency)

## Оглавление

* [1. Входные сущности](#1-входные-сущности)
* [2. Построение BlockGraph из очередей](#2-построение-blockgraph-из-очередей)
* [3. Мультиагентное планирование](#3-мультиагентное-планирование)
* [4. Валидация результата](#4-валидация-результата)
* [5. Пример запуска](#5-пример-запуска)

---

## 1. Входные сущности

* `WorkGraph`: блок регламентных работ.
* `Scheduler`: планировщик агента.
* `Contractor`: ресурсы агента.
* Объекты multi-agency: `Agent`, `Manager`, `ScheduledBlock`.&#x20;

---

## 2. Построение BlockGraph из очередей

Функция `load_queues_bg(queues)` превращает список очередей графов в `BlockGraph` и генерирует зависимости между стартовыми узлами соседних очередей. Шаги: собрать все `WorkGraph`, создать `BlockGraph.pure`, затем добавить рёбра между стартами по шаблону «текущая ↔ предыдущая очередь». Возврат — готовый `BlockGraph`.&#x20;

```python
from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.schemas.graph import WorkGraph

def load_queues_bg(queues: list[list[WorkGraph]]) -> BlockGraph:
    wgs = [wg for queue in queues for wg in queue]
    bg = BlockGraph.pure(wgs)

    index = 0
    nodes_prev = []
    for queue in queues:
        nodes = [bg[wgs[i].start.id] for i in range(index, index + len(queue))]

        # связи между очередями
        for i, node in enumerate(nodes[:-2]):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[i])

        for i, node in enumerate(nodes):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[-i])

        nodes_prev = nodes

    return bg
```

---

## 3. Мультиагентное планирование

`run_example(queues_with_obstructions, schedulers, contractors)` создаёт агентов, строит `BlockGraph`, планирует блоки через `Manager.manage_blocks`, возвращает `Dict[str, ScheduledBlock]`. Объекты препятствий (`Obstruction`, `OneInsertObstruction`) доступны для сценариев, но в базовом запуске не используются.&#x20;

```python
from typing import Dict
from sampo.scheduler.base import Scheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager, ScheduledBlock

def run_example(
    queues_with_obstructions: list[list[WorkGraph]],
    schedulers: list[Scheduler],
    contractors: list[Contractor]
) -> Dict[str, ScheduledBlock]:

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    bg = load_queues_bg(queues_with_obstructions)
    blocks_schedules = manager.manage_blocks(bg, logger=print)
    return blocks_schedules
```

---

## 4. Валидация результата

После планирования проверяется корректность расписания блоков: соответствие зависимостям `BlockGraph` и агентам.&#x20;

```python
from sampo.scheduler.multi_agency import validate_block_schedule

validate_block_schedule(bg, blocks_schedules, agents)
```

---

## 5. Пример запуска

Минимальный сценарий с двумя очередями, двумя агентами и HEFT-планировщиками.

```python
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg

# 1) подготовка очередей графов
ss = SimpleSynthetic(rand=31)
q1 = [ss.small_work_graph() for _ in range(2)]
q2 = [ss.small_work_graph() for _ in range(2)]
queues = [q1, q2]

# 2) планировщики и подрядчики (по одному на агента)
schedulers = [HEFTScheduler(), HEFTScheduler()]
contractors = [get_contractor_by_wg(q1[0]), get_contractor_by_wg(q2[0])]

# 3) мультиагентное планирование
blocks_schedules = run_example(queues, schedulers, contractors)

# 4) печать ключей запланированных блоков
print(list(blocks_schedules.keys()))
```
