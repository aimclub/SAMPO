from sampo.generator.base import SimpleSynthetic
from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg, ContractorGenerationMethod
from sampo.generator import SyntheticGraphType
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.utils.critical_path import critical_path_graph
from sampo.schemas.time_estimator import DefaultWorkEstimator

scheduler = HEFTScheduler()

r_seed = 231
ss = SimpleSynthetic(r_seed)

simple_wg = ss.work_graph(mode=SyntheticGraphType.GENERAL,
                          cluster_counts=10,
                          bottom_border=100,
                          top_border=200)

contractors = [get_contractor_by_wg(simple_wg, 10, ContractorGenerationMethod.MAX)]

project = SchedulingPipeline.create() \
    .wg('9-1-ukpg-full-with-priority.csv', sep=';', all_connections=True) \
    .contractors((ContractorGenerationMethod.MAX, 1, 1000)) \
    .schedule(scheduler) \
    .finish()[0]

schedule = project.schedule

graph_cp = critical_path_graph(project.wg.nodes, work_estimator=DefaultWorkEstimator())

schedule_cp = project.critical_path()
