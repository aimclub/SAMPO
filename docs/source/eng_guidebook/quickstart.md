# Quick Start

This section shows how to install the package, prepare a simple example, run the scheduler, and view the result.

## Installation

SAMPO is available as a Python package (requires Python 3.10.x):

```bash
pip install sampo
```

## First Plan in a Few Steps

Let's create a simple project and define it step by step:

1) **Work graph** — create a `WorkGraph`. For a quick start, we will generate a synthetic one.  
2) **Resources** — define a list of `Contractor` objects with workers.  
3) **Algorithm** — choose a `Scheduler` (heuristic or genetic).  
4) **Run** — generate a `Schedule` and review the result.


---

### Create a WorkGraph (quick method — synthetic generator)

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType
from sampo.schemas.graph import WorkGraph

# Initialize the synthetic generator
synthetic = SimpleSynthetic()

# Generate a small graph: ~2 clusters with 5–8 tasks each
work_graph: WorkGraph = synthetic.work_graph(
    mode=SyntheticGraphType.GENERAL,  # structure type: GENERAL / PARALLEL / SEQUENTIAL
    cluster_counts=2,  # 2 clusters
    bottom_border=5,  # 5–8 tasks per cluster
    top_border=8
)
print(f"Generated a WorkGraph with {len(work_graph.nodes)} tasks.")
```

---

### Resources (Contractors)

**Important:** the synthetic graph uses the following standard job types:  
`driver`, `fitter`, `manager`, `handyman`, `electrician`, `engineer`.

A contractor must include workers for every required job type.  
The `workers` dictionary is keyed by the resource type name (`req.kind`).

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

# Define several workers for each required type
workers = [
    Worker(id="w_driver", name="driver", count=20),
    Worker(id="w_fitter", name="fitter", count=20),
    Worker(id="w_manager", name="manager", count=10),
    Worker(id="w_handyman", name="handyman", count=20),
    Worker(id="w_electrician", name="electrician", count=10),
    Worker(id="w_engineer", name="engineer", count=10),
]

# A single contractor with a complete worker pool
contractors = [
    Contractor(
        id="c1",
        name="General Contractor",
        # Keys are resource type names (match WorkerReq.kind)
        workers={w.name: w for w in workers}
    )
]
```

Alternative: auto-generate a contractor “from the graph” to ensure resource coverage:

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractors = [get_contractor_by_wg(
    work_graph,
    scaler=1.0,  # capacity multiplier (>= 1.0)
    method=ContractorGenerationMethod.AVG,  # average between min/max needs
    contractor_id="c1",
    contractor_name="General Contractor"
)]
```

---

### Choose a Scheduler

```python
from sampo.scheduler.heft import HEFTScheduler

# also available:
# from sampo.scheduler.topological import TopologicalScheduler
# from sampo.scheduler.genetic import GeneticScheduler

scheduler = HEFTScheduler()  # fast heuristic for a quick start
```

---

### Run the Scheduling

The `schedule(...)` method returns a list of `Schedule` objects.  
Take the first one (the best solution):

```python
best_schedule = scheduler.schedule(work_graph, contractors)[0]
print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

If you need additional details (finish time, timeline, node order), use the extended method:

```python
best_schedule, finish_time, timeline, node_order = scheduler.schedule_with_cache(work_graph, contractors)[0]
print(f"Makespan: {best_schedule.execution_time}")
```

---

### View the Schedule (Gantt Chart)

A reliable way is to get an aggregated representation and visualize it:

```python
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

merged = best_schedule.merged_stages_datetime_df(offset='2025-01-01')

fig = schedule_gant_chart_fig(
    merged,
    visualization=VisualizationMode.ReturnFig
    # Specifies what to do with the figure (here: return the object so it can be shown)
)
fig.show()
```

- If you want to see only your tasks without the “internal” technical ones, use the table `best_schedule.pure_schedule_df`.
- For a Gantt chart, it is usually better to use the full calendar representation from `best_schedule.merged_stages_datetime_df`.

---

### (Optional) SchedulingPipeline

The same steps can be performed in a **fluent style** using the `SchedulingPipeline`.  
The `finish()` method returns a list of `ScheduledProject`; take `[0]` and read its `project.schedule`.


```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft import HEFTScheduler

project = (SchedulingPipeline.create()
.wg(work_graph)
.contractors(contractors)
.schedule(HEFTScheduler())
.finish()[0])

print(f"Project duration: {project.schedule.execution_time}")
```
