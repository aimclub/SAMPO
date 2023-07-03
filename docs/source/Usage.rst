How to Use
==========


To use the API, follow these steps:

1. Input data preparation

To use SAMPO for the schedule generation you need to prepare:

* WorkGraph object with the works information representation, including volumes of the works and connections between them
* list of Contractor objects with the information about available resources types and volumes.

    1.1. Loading WorkGraph from file

    .. code-block:: python

      wg = WorkGraph.load(...)

    1.2. Generating synthetic WorkGraph

    .. code-block:: python

      from sampo.generator import SimpleSynthetic

      # SimpleSynthetic object used for the simple work graph structure generation
      ss = SimpleSynthetic()

      # simple graph
      # should generate general (average) type of graph with 10 clusters from 100 to 200 vertices each
      wg = ss.work_graph(mode=SyntheticGraphType.General,
                         cluster_counts=10,
                         bottom_border=100,
                         top_border=200)

      # complex graph
      # should generate general (average) type of graph with 300 unique works, 100 resources and 2000 vertices
      wg = ss.advanced_work_graph(works_count_top_border=2000,
                                  uniq_works=300,
                                  uniq_resources=100)

    1.3. Contractors generation

    Manual Contractor list generation:

    .. code-block:: python

    contractors = [Contractor(id="OOO Berezka", workers=[Worker(id='0', kind='general', count=100)])]


2. Scheduling process

    2.1. Scheduler constructing

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

    2.2. Schedule generation

    .. code-block:: python

      schedule = scheduler.schedule(wg, contractors)

3. Pipeline structure

When data was prepared and scheduler built, you should use scheduling pipeline to control the scheduling process:

.. code-block:: python

  from sampo.pipeline import SchedulingPipeline

  schedule = SchedulingPipeline.create() \
        .wg(wg) \
        .contractors(contractors) \
        .schedule(HEFTScheduler()) \
        .finish()