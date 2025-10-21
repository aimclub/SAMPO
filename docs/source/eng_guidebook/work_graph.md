# WorkGraph

## Terms

- **WorkGraph** — a "project map" represented as a directed acyclic graph. Two service nodes — start and end — are added
  automatically.
- **GraphNode** — a graph node that contains a task (`WorkUnit`) and links to its parent/child nodes (each link has a
  type and a lag).
- **WorkUnit** — the description of a task: its volume and resource requirements (what kinds of workers and how many).

---

## How to Create a WorkGraph

### Generate Automatically (Quick Start)

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType

ss = SimpleSynthetic()
wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,  # structure type: General / Parallel / Sequential
    cluster_counts=10,  # number of clusters (groups of tasks)
    bottom_border=100,  # lower limit of the number of tasks
    top_border=200  # upper limit of the number of tasks
)
```

* `mode` — the form of the graph (general, parallel, sequential).
* `cluster_counts` — how many groups of tasks the graph contains.
* `bottom_border` / `top_border` — the range for the number of tasks.

---

### Load from CSV

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

wg = project.wg  # ready-to-use WorkGraph
schedule = project.schedule
```

* `all_connections=True` — try to reconstruct missing dependencies if they are absent in the input data.
* `sep=';'` — column separator in the CSV file.
* `LagOptimizationStrategy` — specifies how lags are processed (you can rely on the "AUTO" option to choose
  automatically).

> If you use `contractor` autogeneration through `get_contractor_by_wg`, **resource columns are optional** —
> only mandatory fields and dependencies are required.

---

## CSV Structure

### Required Columns

| Field           | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `activity_id`   | Unique identifier of the task                                 |
| `activity_name` | Human-readable task name                                      |
| `granular_name` | Short task code or label                                      |
| `volume`        | Work volume (numeric value)                                   |
| `measurement`   | Unit of measurement (e.g., `unit`, `m3`, `pcs`)               |
| `priority`      | Integer for prioritization (set to `0` for all if not needed) |

### Dependencies

Three synchronized lists (their lengths must match):

* `predecessor_ids` — predecessor IDs separated by commas
* `connection_types` — connection types (FS, SS, FF, etc.)
* `lags` — time delays (numeric values)

> An empty cell means **three consecutive separators with no spaces** (`;;;`).

> Dependency example:  
> `predecessor_ids="A,B"`, `connection_types="FS,SS"`, `lags="0,3"`

### Resources (Optional)

| Field        | Format                                                |
|--------------|-------------------------------------------------------|
| `min_req`    | JSON dictionary of minimally required resources       |
| `max_req`    | JSON dictionary of maximum allowed resources          |
| `req_volume` | JSON dictionary of norms/volumes for time calculation |

* Use **valid JSON** with double quotes:  
  `{"welder": 2, "driver": 1}`
* If these columns are missing, the scheduler automatically generates resources (for example, using
  `get_contractor_by_wg`).
* Keys must match `Worker.name`.

---

### Minimal CSV (Without Resources)

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags
A;Task A;A;1.0;unit;0;;;
B;Task B;B;1.0;unit;0;A;FS;0
C;Task C;C;1.0;unit;0;B;FS;0
```

### CSV with Resources

```
activity_id;activity_name;granular_name;volume;measurement;priority;predecessor_ids;connection_types;lags;min_req;max_req;req_volume
A;Site prep;A;1.0;unit;0;;;;"{""driver"":1,""handyman"":1}";"{""driver"":2,""handyman"":2}";"{""driver"":8,""handyman"":8}"
B;Foundation;B;1.0;unit;0;A;FS;0;"{""fitter"":2,""engineer"":1}";"{""fitter"":4,""engineer"":2}";"{""fitter"":12,""engineer"":12}"
```

---

### Auto-generate a `Contractor` for a CSV-based `WorkGraph`

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

### Create a `WorkGraph` with Python

```python
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time

# Example chain: A -> B -> C (FS, zero lags)
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

* Mandatory `WorkUnit` fields: `id`, `name`.  
* For export, it is also useful to specify `volume`, `measurement`, `priority`.
