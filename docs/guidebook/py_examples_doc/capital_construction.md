# Мультиагентное планирование капстроительства

## Оглавление

* [1. Назначение и сущности](#1-назначение-и-сущности)
* [2. Построение BlockGraph: `load_queues_bg`](#2-построение-blockgraph-load_queues_bg)
* [3. Запуск стохастического мультиагентного планирования: `run_example`](#3-запуск-стохастического-мультиагентного-планирования-run_example)
* [4. Мини-пример использования](#4-мини-пример-использования)

---

## 1. Назначение и сущности

Файл задаёт две ключевые функции: сборку `BlockGraph` из очередей `WorkGraph` и стохастическое мультиагентное планирование блоков с учётом производительности работников по подрядчикам. Используются: `Scheduler`, `Agent`, `StochasticManager`, `ScheduledBlock`, `IntervalGaussian`, `DefaultWorkEstimator`.&#x20;

---

## 2. Построение BlockGraph: `load_queues_bg`

**Идея.** Развернуть список очередей графов в единый `BlockGraph` и сгенерировать зависимости между стартовыми узлами соседних очередей (прямой и «обратной» проход для покрытия недостающих рёбер).&#x20;

```python
from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.schemas.graph import WorkGraph

def load_queues_bg(queues: list[list[WorkGraph]]):
    wgs: list[WorkGraph] = [wg for queue in queues for wg in queue]
    bg = BlockGraph.pure(wgs)

    index = 0
    nodes_prev = []
    for queue in queues:
        nodes = [bg[wgs[i].start.id] for i in range(index, index + len(queue))]

        # прямой проход
        for i, node in enumerate(nodes[:-2]):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[i])

        # обратный проход для покрытия оставшихся
        for i, node in enumerate(nodes):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[-i])

        nodes_prev = nodes

    return bg
```



---

## 3. Запуск стохастического мультиагентного планирования: `run_example`

**Шаги.**

1. Создать `DefaultWorkEstimator`.
2. Для каждого подрядчика `contractor[i]` задать производительность ролей `['driver','fitter','manager','handyman','electrician','engineer']` как `IntervalGaussian(0.2 * i + 0.2, 1, 0, 2)`.
3. Назначить оценщик всем планировщикам.
4. Собрать агентов `Agent(name, scheduler, [contractor])`.
5. Построить `BlockGraph` через `load_queues_bg`.
6. Запустить `StochasticManager.manage_blocks`.&#x20;

```python
from typing import Dict
from sampo.scheduler.base import Scheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, ScheduledBlock, StochasticManager
from sampo.schemas import IntervalGaussian
from sampo.schemas.contractor import Contractor
from sampo.schemas.time_estimator import DefaultWorkEstimator

def run_example(
    queues: list[list[WorkGraph]],
    schedulers: list[Scheduler],
    contractors: list[Contractor]
) -> Dict[str, ScheduledBlock]:
    work_estimator = DefaultWorkEstimator()

    # роль-специфичная производительность по подрядчику i
    for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']:
        for i, contractor in enumerate(contractors):
            work_estimator.set_worker_productivity(
                IntervalGaussian(0.2 * i + 0.2, 1, 0, 2), worker, contractor.id
            )

    for scheduler in schedulers:
        scheduler.work_estimator = work_estimator

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = StochasticManager(agents)

    bg = load_queues_bg(queues)
    blocks_schedules = manager.manage_blocks(bg, logger=print)
    return blocks_schedules
```



---

## 4. Мини-пример использования

```python
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg

# очереди из малых графов
ss = SimpleSynthetic(rand=31)
queues = [[ss.small_work_graph() for _ in range(2)],
          [ss.small_work_graph() for _ in range(2)]]

# планировщики и подрядчики
schedulers = [HEFTScheduler(), HEFTScheduler()]
contractors = [get_contractor_by_wg(queues[0][0]), get_contractor_by_wg(queues[1][0])]

# запуск
blocks = run_example(queues, schedulers, contractors)
print(list(blocks.keys()))
```
