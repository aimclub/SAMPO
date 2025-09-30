# Планирование

## Что такое планировщик (Scheduler)

Планировщик — это объект, который по графу работ (`WorkGraph`) и ресурсам (список `Contractor`) строит итоговый план (
`Schedule`).

Обычно вы просто создаёте нужный планировщик (например, `HEFTScheduler`, `GeneticScheduler`, `TopologicalScheduler`) и
вызываете `.schedule(wg, contractors)`.

Импорты (часто используемые):

```python
from sampo.scheduler.heft import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.topological import TopologicalScheduler
from sampo.scheduler.genetic import GeneticScheduler
```

Мини‑пример:

```python
# есть: wg (WorkGraph), contractors (list[Contractor])
from sampo.scheduler.heft import HEFTScheduler

scheduler = HEFTScheduler()
schedule = scheduler.schedule(wg, contractors)[0]
print("Время проекта:", schedule.execution_time)
```

Через «конвейер» (pipeline):

```python
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import GeneticScheduler

result = (SchedulingPipeline.create()
          .wg('input.csv', sep=';')
          .schedule(GeneticScheduler(number_of_generation=10))
          .finish())[0]

print("Время проекта:", result.schedule.execution_time)
```

### Настройки планировщика

- Как считать длительности работ: передайте свой `work_estimator` (например, если у вас особая формула времени).
- Как подбирать размер бригад/ресурсов: передайте другой `resource_optimizer` (например, пробовать больше/меньше людей).
- Свой алгоритм: сделайте класс‑наследник и задайте `scheduler_type` для внутренней идентификации.

---

## Что такое Schedule

`Schedule` — это весь итоговый план:

- `execution_time` — длительность проекта (от первого старта до последнего финиша),
- готовые представления (таблицы, диаграммы),
- экспорт/фильтрация, служебные операции,
- (опционально) порядок выполнения, статистика по ресурсам, «критический путь».

Пример:

```python
schedule = scheduler.schedule(wg, contractors)[0]
print("Время проекта:", schedule.execution_time)

df = schedule.merged_stages_datetime_df(start_date="2025-01-01")
print(df.head())
```

---