from random import Random

from sampo.scheduler import GeneticScheduler, TopologicalScheduler, RandomizedTopologicalScheduler, LFTScheduler, \
    RandomizedLFTScheduler
from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
from sampo.generator import SyntheticGraphType
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.schedule_spec import ScheduleSpec

ss = SimpleSynthetic(rand=231)
rand = Random()
size = 100

wg = ss.work_graph(bottom_border=size - 5, top_border=size)

contractors = [get_contractor_by_wg(wg) for _ in range(10)]
spec = ScheduleSpec()

for node in wg.nodes:
    if not node.is_inseparable_son():
        selected_contractor_indices = rand.choices(list(range(len(contractors))),
                                                   k=rand.randint(1, len(contractors)))
        spec.assign_contractors(node.id, {contractors[i].id for i in selected_contractor_indices})


scheduler = GeneticScheduler(number_of_generation=10)

project = SchedulingPipeline.create() \
    .wg(wg) \
    .contractors(contractors) \
    .lag_optimize(LagOptimizationStrategy.TRUE) \
    .spec(spec) \
    .schedule(scheduler, validate=True) \
    .visualization('2022-01-01')[0] \
    .shape((14, 14)) \
    .show_gant_chart()
