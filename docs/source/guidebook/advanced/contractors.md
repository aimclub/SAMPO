# Contractor

## Термины

- Требование к людям (WorkerReq) — кого и сколько нужно для задачи: профессия (`kind`), минимум/максимум людей (
  `min_count…max_count`), и объём/норма (`volume`).
- Работник (Worker) — что у нас есть «на складе»: специализация (`name`), количество (`count`), производительность (
  `productivity`), владелец (`contractor_id`), при необходимости — стоимость.
    - Важно: `Worker.name` должен совпадать с `WorkerReq.kind` из задач — так система понимает, что это подходящий
      ресурс.
- Техника (Equipment) — тоже ресурс, только не люди: тип и количество. Задачи используют её так же, как и людей.
- Подрядчик (Contractor) — поставщик ресурсов. По сути — словарь доступных «людей» и «техники» для планировщика.

---

## Как задать подрядчика

### Вариант 1. Вручную

```python
from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.interval import IntervalGaussian

# Пример: подрядчик с двумя профессиями (водители и слесари)
contractor = Contractor(
    workers={
        # ключ словаря 'driver' совпадает с Worker.name='driver'
        'driver': Worker(
            id='w1',
            name='driver',  # профессия (должна совпадать с WorkerReq.kind)
            count=8,  # сколько таких людей доступно
            productivity=IntervalGaussian(1.0, 0.1, 0.5, 1.5)  # «средняя скорость» с разбросом
            # cost_one_unit=...,    # при расчёте стоимости — укажите цену за единицу (опционально)
            # contractor_id='c1',   # как правило, совпадает с Contractor.id (если задаёте вручную)
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

Подсказки:

- Ключи словаря `workers` должны совпадать с `Worker.name`.
- `WorkerReq.kind` из задач должен совпадать с `Worker.name`, иначе бригада не подберётся.
- `contractor_id` у работников обычно совпадает с `Contractor.id`.
- `cost_one_unit` укажите, если хотите считать деньги, иначе можно опустить.
- `IntervalGaussian(μ, σ, low, high)` — «средняя производительность μ», с разбросом ±σ и ограничениями `low…high`.

---

### Вариант 2. Быстрый генератор «пакета» ресурсов

```python
from sampo.generator.base import SimpleSynthetic

ss = SimpleSynthetic()
contractor = ss.contractor(pack_worker_count=10)
```

Что это даёт:

- Один «универсальный» подрядчик с типовым набором работников.
- Параметр `pack_worker_count` задаёт «размер пакета» — грубо, среднее количество людей на каждый востребованный тип.

Когда полезно:

- Когда нужно быстро запустить примеры/эксперименты без ручного ввода ресурсов.

---

### Вариант 3. Подрядчик «по графу» (из требований задач)

```python
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod

# wg — ваш WorkGraph с задачами и их WorkerReq
contractor = get_contractor_by_wg(
    wg,
    scaler=1.0,  # можно умножить ресурсы (1.5 = +50%)
    method=ContractorGenerationMethod.AVG  # агрегировать требования «в среднем»
)
```

Как это работает:

- Смотрит на все `WorkerReq` в проекте.
- Складывает/усредняет потребности по профессиям.
- Собирает подрядчика с подходящим количеством людей/типов.
- `scaler` позволяет быстро «дать больше/меньше» ресурсов, не переписывая всё вручную.

---

---

## Ещё пара мини‑примеров

Два подрядчика раздельно:

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

Быстро «нарастить» ресурсы у существующего:

```python
# Было 6 водителей, станет 10
contractor.workers['driver'].count = 10
```

---

Коротко:

- Contractor — это «корзина» ресурсов: кто у нас есть и в каком количестве.
- Главное правило соответствия: `WorkerReq.kind` (в задачах) = `Worker.name` (у подрядчиков).
- Начните с простого: сгенерируйте подрядчика автоматически, а потом подправьте руками узкие места.
