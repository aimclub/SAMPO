# Contractor

## Terms

- **Worker Requirement (`WorkerReq`)** — defines who and how many people are needed for a task:  
  profession (`kind`), minimum/maximum number of workers (`min_count…max_count`), and work volume or productivity rate (
  `volume`).
- **Worker** — represents human resources: specialization (`name`), quantity (`count`), productivity (`productivity`),  
  owner (`contractor_id`), and optionally cost per unit (`cost_one_unit`).
- **Equipment** — a non-human resource: its type and quantity. Tasks use it the same way as workers.
- **Contractor** — a resource provider, essentially a dictionary of available workers and equipment for the scheduler.

---

## Available Professions:

- 'driver'
- 'fitter'
- 'handyman'
- 'electrician'
- 'manager'
- 'engineer'

## How to Define a Contractor

### Manually

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

# Example: a contractor with two professions (drivers and fitters)
contractor = Contractor(
    workers={
        # dictionary key 'driver' must match Worker.name='driver'
        'driver': Worker(
            id='w1',
            name='driver',  # profession (must match WorkerReq.kind)
            count=8,  # number of available workers
            productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)  # average productivity with deviation
            # cost_one_unit=... optional: cost per unit if cost calculation is required
            # contractor_id='c1' usually matches Contractor.id (if set manually)
        ),
        'fitter': Worker(
            id='w2',
            name='fitter',
            count=6,
            productivity=IntervalGaussian(1.2, 0.1, 0.8, 1.6)
        ),
    },
    id='c1',
    name='Contractor A'
)
```

**Notes:**

* Dictionary keys in `workers` **must match** each `Worker.name`.
* The field `WorkerReq.kind` from tasks must match `Worker.name`; otherwise, the scheduler cannot assign workers.
* The `contractor_id` of each worker usually matches `Contractor.id`.
* `IntervalGaussian(μ, σ, low, high)` — defines **average productivity (μ)** with standard deviation (σ) and range
  limits `low…high`.

### Quick resource package generator

```python
from sampo.generator.base import SimpleSynthetic

ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
# pack_worker_count: A scaling factor for the number of workers in each profession
```

---

### Contractor “from Graph” (based on task requirements)

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

# wg — your WorkGraph with tasks and their WorkerReq
contractor = get_contractor_by_wg(
    wg,
    scaler=1.0,  # scale resource capacities (e.g., 1.5 = +50%)
    method=ContractorGenerationMethod.AVG  # aggregate requirements by average
)
```

**How it works:**

* Analyzes all `WorkerReq` in the project.
* Aggregates or averages resource needs by job type.
* Builds a contractor with an appropriate number of workers per profession.
* The `scaler` parameter lets you quickly increase or decrease available resources without editing them manually.

---

## A Few More Mini-Examples

Two contractors defined separately:

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker

contractor_a = Contractor(
    id='ca', name='Team A',
    workers={'welder': Worker(id='wa', name='welder', count=4)}
)

contractor_b = Contractor(
    id='cb', name='Team B',
    workers={'driver': Worker(id='wb', name='driver', count=6)}
)

contractors = [contractor_a, contractor_b]
```

Quickly increase resources for an existing one:

```python
# Previously there were 6 drivers; now there will be 10
contractor.workers['driver'].count = 10
```

---