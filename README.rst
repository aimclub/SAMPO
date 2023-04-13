.. image:: docs/sampo_logo.png
   :alt: Logo of SAMPO framework
   :width: 300pt
   
Scheduler for Adaptive Manufacturing Processes Optimization
======================

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |py_10|
   * - license
     - | |license|
   * - support
     - | |mailto|


.. end-badges

**SAMPO** is an open-source framework for adaptive manufacturing processes scheduling. This framework is distributed under the 3-Clause BSD license.

It provides toolbox for generating schedules of manufacturing process under the constraints imposed by the subject area. The core of SAMPO is based on the combination of meta-heuristic, genetic and multi-agent algorithms. Taking as input the task graph with tasks connections and resource constraints, as well as the optimization metric, the scheduler builds the optimal tasks sequence and resources assignment according to the given metric.


Installation
============

SAMPO package is available via PyPI:

.. code-block::

  $ pip install sampo

SAMPO Features
============

The following algorithms for projects sheduling are implemented:

* Topological - heuristic algorithm based in toposort of WorkGraph
* HEFT (heterogeneous earliest finish time) and HEFTBetween - heuristic algorithms based on critical path heuristic
* Genetic - algorithm that uses heuristic algorithms for beginning population and modelling evolution process

Difference from existing implementations:

* Module for generating graphs of production tasks with a given structure;
* Easy to use pipeline structure;
* Multi-agent modeling block, allowing you to effectively select a combination of planning algorithms for a particular project;
* Ability to handle complex projects with a large number of works (2-10 thousand).

How to Use
==========


To use the API, follow these steps:

1. Prepare data

To use SAMPO as scheduler you need WorkGraph as work info representation and list of Contractor
objects as available resources.

    1.1. Load WorkGraph from file

    .. code-block:: python

      wg = WorkGraph.load(...)

    1.2. Generate synthetic WorkGraph

    .. code-block:: python

      from sampo.generator import SimpleSynthetic

      # SimpleSynthetic object used for simpler generations
      ss = SimpleSynthetic()

      # simple graph
      # should generate general(average) type of graph with 10 clusters and from 100 to 200 vertices
      wg = ss.work_graph(mode=SyntheticGraphType.General,
                         cluster_counts=10,
                         bottom_border=100,
                         top_border=200)

      # complex graph
      # should generate general(average) type of graph with 300 unique works, 100 resources and below 2000 vertices
      wg = ss.advanced_work_graph(works_count_top_border=2000,
                                  uniq_works=300,
                                  uniq_resources=100)

    1.3. Contractors

        1.3.1. Construct by hand

        .. code-block:: python

          contractors = [Contractor(id="OOO Berezka", workers=[Worker(id='0', kind='general', count=100)])]

        1.3.2. Generate from WorkGraph

        .. code-block:: python

          # TODO

2. Schedule

    2.1. Construct the scheduler

    There are 4 classes of schedulers available in SAMPO:

    - HEFTScheduler
    - HEFTBetweenScheduler
    - TopologicalScheduler
    - GeneticScheduler

    Each of them has various hyper-parameters to fit. They should be passed when scheduler object created.

    .. code-block:: python

      from sampo.scheduler.heft import HEFTScheduler

      scheduler = HEFTScheduler()

    .. code-block:: python

      from sampo.scheduler.genetic import GeneticScheduler

      scheduler = GeneticScheduler(mutate_order=0.1,
                                   mutate_resources=0.3)

    2.2. Schedule

    .. code-block:: python

      schedule = scheduler.schedule(wg, contractors)

3. Pipeline

When data was prepared and scheduler built, you should use scheduling pipeline to control the scheduling process:

.. code-block:: python

  from sampo.pipeline import SchedulingPipeline

  schedule = SchedulingPipeline.create() \
        .wg(wg) \
        .contractors(contractors) \
        .schedule(HEFTScheduler()) \
        .finish()

.. |pypi| image:: https://badge.fury.io/py/sampo.svg
   :alt: Supported Python Versions
   :target: https://badge.fury.io/py/sampo


.. |py_10| image:: https://img.shields.io/badge/python_3.10-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.10-passing-success

.. |license| image:: https://img.shields.io/github/license/Industrial-AI-Research-Lab/sampo
   :alt: Supported Python Versions
   :target: https://github.com/Industrial-AI-Research-Lab/sampo/blob/master/LICENSE


.. |mailto| image:: https://img.shields.io/badge/email-IAIRLab-blueviolet
   :alt: Framework Support
   :target: mailto:iairlab@yandex.ru
