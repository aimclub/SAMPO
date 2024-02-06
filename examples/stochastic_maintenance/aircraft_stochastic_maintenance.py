from random import Random

import numpy as np

from sampo.generator import SimpleSynthetic, SyntheticGraphType
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import GeneticScheduler
from sampo.schemas import DefaultZoneStatuses, ZoneConfiguration, LandscapeConfiguration
from sampo.schemas.structure_estimator import DefaultStructureGenerationEstimator, DefaultStructureEstimator
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode

r_seed = 231
ss = SimpleSynthetic(r_seed)

wg = ss.work_graph(mode=SyntheticGraphType.GENERAL,
                   cluster_counts=10,
                   bottom_border=100,
                   top_border=200)

rand = Random(r_seed)
generator = DefaultStructureGenerationEstimator(rand)

sub_works = [f'Sub-work {i}' for i in range(5)]

# assign 5 uniform-distributed generations to each non-service node
for node in wg.nodes:
    if node.work_unit.is_service_unit:
        continue
    for sub_work in sub_works:
        generator.set_probability(parent=node.work_unit.name, child=sub_work, probability=1 / len(sub_works))

structure_estimator = DefaultStructureEstimator(generator, rand)

# perform restructuring
wg_with_defects = structure_estimator.restruct(wg)
contractors = [get_contractor_by_wg(wg_with_defects)]


# creating zone configuration
class AeroplaneZoneStatuses(DefaultZoneStatuses):
    def statuses_available(self) -> int:
        return 4

zone_names = ['zone1']
zones_count = len(zone_names)

zone_config = ZoneConfiguration(
    start_statuses={zone: 1 for zone in zone_names},
    time_costs=np.array([[1 for _ in range(zones_count)] for _ in range(zones_count)]),
    statuses=AeroplaneZoneStatuses()
)
landscape_config = LandscapeConfiguration(zone_config=zone_config)

work_estimator = DefaultWorkEstimator()

genetic_scheduler = GeneticScheduler(work_estimator=work_estimator,
                                     number_of_generation=20,
                                     mutate_order=0.05,
                                     mutate_resources=0.005,
                                     size_of_population=50)

aircraft_project = SchedulingPipeline.create() \
    .wg(wg_with_defects) \
    .contractors(contractors) \
    .work_estimator(work_estimator) \
    .landscape(landscape_config) \
    .schedule(genetic_scheduler) \
    .finish()


merged_schedule = aircraft_project.schedule.merged_stages_datetime_df('2022-01-01')
aircraft_schedule_fig = schedule_gant_chart_fig(merged_schedule, VisualizationMode.ReturnFig, remove_service_tasks=False)
aircraft_schedule_fig.update_layout(height=1200, width=1600)
aircraft_schedule_fig.show()
