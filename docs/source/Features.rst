Features
============

The following algorithms for projects sheduling are implemented:

* Topological - heuristic algorithm based in toposort of WorkGraph
* HEFT (heterogeneous earliest finish time) and HEFTBetween - heuristic algorithms based on critical path heuristic
* Genetic - algorithm that uses heuristic algorithms for beginning population and modelling evolution process

Difference from existing implementations:

* Module for generating graphs of production tasks with a given structure
* Easy to use pipeline structure
* Multi-agent modeling block, allowing you to effectively select a combination of planning algorithms for a particular project
* Ability to handle complex projects with a large number of works (2-10 thousand)