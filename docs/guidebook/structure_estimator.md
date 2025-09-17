# Оценка и восстановление структуры работ (StructureEstimator) в SAMPO

## Оглавление

* [1. Генерация исходного графа](#1-генерация-исходного-графа)

  * [1.1 Синтетический граф](#11-синтетический-граф)
* [2. Создание StructureEstimator](#2-создание-structureestimator)

  * [2.1 Генератор структурных связей](#21-генератор-структурных-связей)
  * [2.2 Итоговый оценщик структуры](#22-итоговый-оценщик-структуры)
* [3. Применение в PreparationPipeline](#3-применение-в-preparationpipeline)

  * [3.1 Реструктуризация WorkGraph](#31-реструктуризация-workgraph)

---

## 1. Генерация исходного графа

### 1.1 Синтетический граф

* Импорт генератора и типов: `SimpleSynthetic`, `SyntheticGraphType`.
* Фиксация зерна для воспроизводимости.
* Построение базового `WorkGraph`.

```python
from sampo.generator.base import SimpleSynthetic
from sampo.generator import SyntheticGraphType  # или: from sampo.generator.pipeline import SyntheticGraphType

r_seed = 231
ss = SimpleSynthetic(r_seed)

wg = ss.work_graph(
    mode=SyntheticGraphType.GENERAL,
    cluster_counts=10,
    bottom_border=100,
    top_border=200,
)
```

## 2. Создание StructureEstimator

### 2.1 Генератор структурных связей

* Используется `DefaultStructureGenerationEstimator`.
* Ему передаётся общий ГПСЧ `Random(r_seed)`.
* Задаются вероятности порождения подработ для каждого несервисного узла.

```python
from random import Random
from sampo.schemas.structure_estimator import DefaultStructureGenerationEstimator

rand = Random(r_seed)
generator = DefaultStructureGenerationEstimator(rand)

sub_works = [f"Sub-work {i}" for i in range(5)]

# равномерно распределяем 5 «подработ» на каждый несервисный узел
for node in wg.nodes:
    if node.work_unit.is_service_unit:
        continue
    for sub_work in sub_works:
        generator.set_probability(
            parent=node.work_unit.name,
            child=sub_work,
            probability=1 / len(sub_works),
        )
```

### 2.2 Итоговый оценщик структуры

* Обёртка `DefaultStructureEstimator(generator, rand)` для использования в пайплайне.

```python
from sampo.schemas.structure_estimator import DefaultStructureEstimator

structure_estimator = DefaultStructureEstimator(generator, rand)
```

## 3. Применение в PreparationPipeline

### 3.1 Реструктуризация WorkGraph

* Подключаем `structure_estimator` в `PreparationPipeline`.
* Строим обновлённый `WorkGraph` с учётом сгенерированных подработ и связей.

```python
from sampo.pipeline.preparation import PreparationPipeline

restructed_wg = (
    PreparationPipeline()
    .wg(wg)
    .structure_estimator(structure_estimator)
    .build_wg()
)

restructed_wg.vertex_count  # проверка изменения размера графа
```
