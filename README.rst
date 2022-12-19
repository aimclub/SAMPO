.. image:: docs/sampo_logo.png
   :alt: Logo of SAMPO framework
   
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

  wg = generator.SimpleSynthetic().advanced_work_graph(works_count_top_border=2000, uniq_works=300, uniq_resources=100)
  contractors = [generator.get_contractor_by_wg(wg)]
  
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


.. |mailto| image:: https://img.shields.io/badge/email-IAIRLab-blueviolet
   :alt: Framework Support
   :target: mailto:iairlab@yandex.ru
