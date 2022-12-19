SAMPO — Scheduler for Adaptive Manufacturing Processes Optimization
==============

## Описание планировщика

Планировщик для адаптивной оптимизации производственных процессов включает в себя набор алгоритмов интеллектуального анализа и построения расписаний задач производственных процессов с учетом ресурсных и прочих ограничений, накладываемых предметной областью.

Он позволяет эффективно планировать производственные задачи и назначать ресурсы, оптимизируя результат планирования по требуемым метрикам.

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |py_10|
   * - license
     - | |license|
   * - support
     - | iairlab@yandex.ru


.. end-badges

**SAMPO** is an open-source framework for adaptive manufacturing processes scheduling. This framework is distributed under the 3-Clause BSD license.



SAMPO Features
==============

The main features of the framework are follows:


Installation
============

The simplest way to install SAMPO is using ``pip``:

.. code-block::

  $ pip install sampo

How to Use
==========


To use the API, follow these steps:

1. Import ``generator`` and ``scheduler`` modules

.. code-block:: python

 from sampo import generator
 from sampo import scheduler

2. Generate synthetic graph and schedule works

.. code-block:: python

  srand = generator.SimpleSynthetic()
  wg = srand.advanced_work_graph(works_count_top_border=2000, uniq_works=300, uniq_resources=100)
  contractors = [get_contractor_by_wg(wg)]
  
  schedule = scheduler.HEFTScheduler().schedule(wg, contractors)
  
  
.. |pypi| image:: https://badge.fury.io/py/sampo.svg
   :alt: Supported Python Versions
   :target: https://badge.fury.io/py/sampo


.. |py_10| image:: https://img.shields.io/badge/python_3.10-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.10-passing-success

.. |license| image:: https://img.shields.io/github/license/Industrial-AI-Research-Lab/sampo
   :alt: Supported Python Versions
   :target: https://github.com/Industrial-AI-Research-Lab/sampo/blob/master/LICENSE
