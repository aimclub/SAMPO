from sampo.scheduler import GeneticScheduler, TopologicalScheduler, RandomizedTopologicalScheduler, LFTScheduler, \
    RandomizedLFTScheduler
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
from sampo.generator import SyntheticGraphType
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler

scheduler = GeneticScheduler()

project = SchedulingPipeline.create() \
    .wg('9-1-ukpg-full-with-priority.csv', sep=';', all_connections=True) \
    .lag_optimize(LagOptimizationStrategy.TRUE) \
    .schedule(scheduler) \
    .visualization('2022-01-01')[0] \
    .shape((14, 14)) \
    .color_type('priority') \
    .show_gant_chart()

schedule = project.schedule
