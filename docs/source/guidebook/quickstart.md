# Quick Start

We will show installation, preparation of a simple example, running the scheduler, and viewing the result.

## Installation

SAMPO is available as a Python package (Python 3.10.x required):

```bash
pip install sampo
```

## First plan in a few steps

We will create the simplest project and lay it out:

1) Work graph — create a WorkGraph. For a quick start, we will generate a synthetic one.
2) Resources — describe a list of Contractors with workers.
3) Algorithm — choose a scheduler (heuristic/genetic).
4) Run — get a Schedule and look at the result.

---

### 1) Generating a WorkGraph (quick way — generator)

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

### 2) Resources (Contractors)

Important: the synthetic graph uses typical professions `driver`, `fitter`, `manager`, `handyman`, `electrician`,
`engineer`.  
The contractor must include workers for each required type, and the `workers` dictionary is keyed by the resource kind
name (`req.kind`).

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

# Specify several workers of each required type
workers = [
    Worker(id="w_driver", name="driver", count=20),
    Worker(id="w_fitter", name="fitter", count=20),
    Worker(id="w_manager", name="manager", count=10),
    Worker(id="w_handyman", name="handyman", count=20),
    Worker(id="w_electrician", name="electrician", count=10),
    Worker(id="w_engineer", name="engineer", count=10),
]

# One contractor with a complete pool of workers
contractors = [
    Contractor(
        id="c1",
        name="General Contractor",
        # Keys are resource kind names (match WorkerReq.kind)
        workers={w.name: w for w in workers}
    )
]
```

Alternative: auto-generate a contractor “based on the graph” so that resources exactly cover the requirements:

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractors = [get_contractor_by_wg(
    work_graph,
    scaler=1.0,  # capacity multiplier (>= 1.0)
    method=ContractorGenerationMethod.AVG,  # averaging between min/max needs
    contractor_id="c1",
    contractor_name="General Contractor"
)]
```

---

### 3) Choosing a scheduler

```python
from sampo.scheduler.heft import HEFTScheduler

# also available:
# from sampo.scheduler.topological import TopologicalScheduler
# from sampo.scheduler.genetic import GeneticScheduler

scheduler = HEFTScheduler()  # fast heuristic to get started
```

---

### 4) Running the scheduling

The schedule(...) method returns a list of Schedule objects. Take the first (best) solution:

```python
best_schedule = scheduler.schedule(work_graph, contractors)[0]
print(f"Projected project duration (makespan): {best_schedule.execution_time}")
```

If you need additional information (finish time, timeline, node order), use the extended method:

```python
best_schedule, finish_time, timeline, node_order = scheduler.schedule_with_cache(work_graph, contractors)[0]
print(f"Makespan: {best_schedule.execution_time}")
```

---

### Viewing the schedule (Gantt chart)

A reliable way is to obtain an aggregated representation and visualize it:

```python
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

merged = best_schedule.merged_stages_datetime_df(offset='2025-01-01')

fig = schedule_gant_chart_fig(
    merged,
    visualization=VisualizationMode.ReturnFig,
    color_type='contractor'  # you can change the coloring if needed
)
fig.show()
```

- If you want to see only your tasks without “internal” technical ones, use the table best_schedule.pure_schedule_df.
- For a Gantt chart, it is common to use the full calendar representation best_schedule.merged_stages_datetime_df.

---

### 5) (Optional) SchedulingPipeline

An equivalent of the same steps in a “fluent” style. finish() returns a list of ScheduledProject; take [0] and read
project.schedule.

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