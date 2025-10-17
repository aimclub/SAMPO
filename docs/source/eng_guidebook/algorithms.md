# Scheduling Algorithms

## How to choose an algorithm

### Heuristic schedulers (HEFT, HEFTBetween, Topological)

- Provide a workable schedule quickly.
- HEFT/HEFTBetween compute task order and approximate execution times to reduce project makespan.
- Topological orders tasks by dependencies without complex optimization.
- HEFT builds a plan from scratch over the entire graph, while HEFTBetween inserts new activities into an already
  occupied calendar, trying not to reshuffle the previously built plan.

Imports:

```python
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.heft import HEFTBetweenScheduler
from sampo.scheduler.topological import TopologicalScheduler
```

---

### Genetic scheduler

- Tries many different combinations and often finds a schedule with a shorter project completion time (minimizing project makespan).  
  Works slower than simpler algorithms.
- Key parameters:
    - `number_of_generation` — how many times to improve the solutions (more → higher chance of improvement, but slower).
    - `size_of_population` — how many alternatives to keep simultaneously (more → more ideas, but slower and more memory-consuming).
    - `mutate_order` — how often to change the task order (while preserving dependencies)  
      (higher values → wider search, but slower convergence).
    - `mutate_resources` — how often to change the allocation of resources/contractors  
      (higher values → better parallelism, but may cause more “conflicts” if resources are scarce).
    - Additionally: `work_estimator`, `seed` (for reproducibility).

Example:

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

## Multi-Agent Scheduling

Splits a project into parts (blocks), schedules them using different strategies, and assembles a global plan.  
Useful for large projects and for combining strategies (`sampo.scheduler.multi_agency`).

### “Auction” without block partitioning

```python
from uuid import uuid4
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

# 1) Small work graph
ss = SimpleSynthetic(231)
wg = ss.work_graph(bottom_border=30, top_border=40)

# 2) Contractor with the required worker types
kinds = {req.kind for node in wg.nodes for req in node.work_unit.worker_reqs}
cid = str(uuid4())
workers = {k: Worker(str(uuid4()), k, 50, contractor_id=cid) for k in kinds}
contractors = [get_contractor_by_wg(
    wg,
    scaler=1.0,  # increase if needed, e.g., 1.2 or 1.5
    method=ContractorGenerationMethod.AVG,
)]

# 3) Two agents with different approaches
agents = [
    Agent("HEFT", HEFTScheduler(), contractors),
    Agent("Topological", TopologicalScheduler(), contractors),
]
manager = StochasticManager(agents)

# 4) “Auction”: agents propose schedules; pick the best by finish time
start, end, schedule, winner = manager.run_auction(wg)
print("Winning agent:", winner.name, "Makespan:", end - start)
```

### With Block Partitioning

```python
from random import Random
from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft import HEFTScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.multi_agency.multi_agency import Agent, StochasticManager
from sampo.scheduler.multi_agency.block_generator import generate_blocks, SyntheticBlockGraphType

# 1) Build a "block graph" (each block is its own small WorkGraph)
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

# 2) Contractors for agents
ss = SimpleSynthetic(rand)
contractor_a = ss.contractor(40)
contractor_b = ss.contractor(40)

# 3) Agents with different algorithms
agents = [
    Agent("HEFT", HEFTScheduler(), [contractor_a]),
    Agent("Topo", TopologicalScheduler(), [contractor_b]),
]
manager = StochasticManager(agents)

# 4) Schedule blocks in order, respecting inter-block dependencies
scheduled_blocks = manager.manage_blocks(bg)

# 5) Summary: who took which block and when it ran
print("Scheduled blocks:")
for block_id, sblock in scheduled_blocks.items():
    print(f"Block {block_id}: agent={sblock.agent.name}, "
          f"start={sblock.start_time}, end={sblock.end_time}, duration={sblock.duration}")

project_finish = max(sb.end_time for sb in scheduled_blocks.values())
print("Project finish time:", project_finish)
```
