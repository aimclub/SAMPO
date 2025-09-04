# Genetic Scheduling API / Генетический API планирования

## Overview / Обзор
Genetic scheduling tools for constructing project plans using evolutionary strategies.
Инструменты генетического планирования для построения графиков проекта с помощью эволюционных стратегий.

## Key Components / Основные компоненты
- `genetic_api.py` – primary interface for building custom genetic schedulers.
  `genetic_api.py` – основной интерфейс для создания собственных генетических планировщиков.

  ```python
  from sampo.api.genetic_api import FitnessFunction, Individual

  class Makespan(FitnessFunction):
      def evaluate(self, chromosome, evaluator):
          schedule = evaluator(chromosome)
          return schedule.makespan,  # minimize project duration // минимизировать длительность проекта

  make_individual = Individual.prepare(Makespan)
  ```

- `const.py` – shared constants across the API.
  `const.py` – общие константы для всего интерфейса.
- `__init__.py` – package initialization.
  `__init__.py` – инициализация пакета.

## Further Reading / Дополнительные материалы
For extended details, see the [main project documentation](../../README.rst).
Для подробностей смотрите [основную документацию проекта](../../README.rst).
