# Contractor

## Термины

**WorkerReq** — требование к рабочей силе: профессия (`kind`), диапазон численности (`min_count`…`max_count`),
объём/норма (`volume`).
> Определяет потребность задачи в людях.

**Worker** — доступный тип рабочей силы: `name` (специализация), `count`, `productivity`, `contractor_id`, опционально
стоимость.
> `Worker.name` сопоставляется с `WorkerReq.kind`.

**Equipment** — единица или группа техники с типом и количеством.
> Используется задачами аналогично рабочим ресурсам.

**Contractor** — поставщик ресурсов (словарь работников, техники и пр.).
> Предоставляет набор доступных `Worker` / `Equipment`.

---

## Подрядчики (Contractor)

### Вручную

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

contractor = Contractor(
    workers={
        'driver': Worker(id='w1', name='driver', count=8, productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)),
        'fitter': Worker(id='w2', name='fitter', count=6, productivity=IntervalGaussian(1.2, 0.1, 0.8, 1.6)),
    },
    id='c1',
    name='Contractor A'
)
```

> Важно:
>
> * Ключи словаря `workers` должны совпадать с `Worker.name`.
> * `WorkerReq.kind` должен совпадать с `Worker.name`, иначе бригада не подберётся.
> * `contractor_id` у `Worker` берётся из `Contractor.id`.
> * При необходимости задайте `cost_one_unit` явно.

### Генератором по параметру «размер пакета»

```python
from sampo.generator.base import SimpleSynthetic

ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
```

### По графу работ (из требований)

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

contractor = get_contractor_by_wg(wg, scaler=1.0, method=ContractorGenerationMethod.AVG)
```

Коротко: агрегирует `min/max` из `WorkerReq` по задачам и формирует пул ресурсов.

---