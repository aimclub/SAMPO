# Contractor

## Terms

- Worker requirement (WorkerReq) — who and how many are needed for a task: profession (kind), minimum/maximum people (
  min_count…max_count), and volume/rate (volume).
- Worker — human resources: specialization (name), count (count), productivity (productivity), owner (contractor_id),
  and optionally cost (cost_one_unit).
- Equipment — also a resource, just not people: type and quantity. Tasks use it the same way as people.
- Contractor — a resource supplier. Essentially, a dictionary of available “people” and “equipment” for the scheduler.

---

## Available professions:

- 'driver'
- 'fitter'
- 'handyman'
- 'electrician'
- 'manager'
- 'engineer'

## How to define a contractor

### Option 1. Manually

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

# Example: contractor with two professions (drivers and fitters)
contractor = Contractor(
    workers={
        # the dictionary key 'driver' matches Worker.name='driver'
        'driver': Worker(
            id='w1',
            name='driver',  # profession (must match WorkerReq.kind)
            count=8,  # how many such people are available
            productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)  # “average rate” with spread
            # cost_one_unit=...,    # when calculating cost — specify unit price (optional)
            # contractor_id='c1',   # usually matches Contractor.id (if set manually)
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

Tips:

- The keys of the workers dictionary must match Worker.name.
- WorkerReq.kind in tasks must match Worker.name, otherwise a crew cannot be assigned.
- contractor_id for workers usually matches Contractor.id.
- IntervalGaussian(μ, σ, low, high) — “average productivity μ”, with spread ±σ and bounds low…high.

---

### Option 2. Quick generator of a resource “pack”

```python
from sampo.generator.base import SimpleSynthetic

ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
```

---

### Option 3. Contractor “based on the graph” (from task requirements)

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

# wg — your WorkGraph with tasks and their WorkerReq
contractor = get_contractor_by_wg(
    wg,
    scaler=1.0,  # you can scale resources (1.5 = +50%)
    method=ContractorGenerationMethod.AVG  # aggregate requirements “on average”
)
```

How it works:

- Looks at all WorkerReq in the project.
- Sums/averages needs by profession.
- Assembles a contractor with a suitable number and types of people.
- scaler lets you quickly “give more/less” resources without rewriting everything manually.

---

## A couple more mini-examples

Two separate contractors:

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

Quickly “scale up” resources for an existing one:

```python
# There were 6 drivers, it will become 10
contractor.workers['driver'].count = 10
```

---