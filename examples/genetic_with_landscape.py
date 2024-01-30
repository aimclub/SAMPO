import random
import uuid
from itertools import chain

import pandas as pd

from sampo.generator import SimpleSynthetic
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import SchedulingPipeline, DefaultInputPipeline
from sampo.scheduler import GeneticScheduler, HEFTBetweenScheduler
from sampo.schemas import LandscapeConfiguration, ResourceHolder, Material, MaterialReq, EdgeType, WorkGraph
from sampo.schemas.landscape import Vehicle
from sampo.schemas.landscape_graph import LandGraphNode, ResourceStorageUnit, LandGraph
from sampo.structurator import graph_restructuring
from sampo.userinput import CSVParser
from sampo.utilities.sampler import Sampler
from sampo.utilities.visualization import VisualizationMode

from field_dev_resources_time_estimator import FieldDevWorkEstimator


if __name__ == '__main__':
    work_time_estimator = FieldDevWorkEstimator()
    wg, contractors = \
        CSVParser.work_graph_and_contractors(
            works_info=CSVParser.read_graph_info(project_info='electroline.csv',
                                                 sep_wg=',',
                                                 all_connections=True,
                                                 change_connections_info=False,
                                                 history_data=pd.DataFrame(columns=['marker_for_glue', 'work_name', 'first_day', 'last_day',
                                                                                    'upper_works', 'work_name_clear_old', 'smr_name',
                                                                                    'work_name_clear', 'granular_smr_name']),
                                                 )
        )
    for node in wg.nodes:
        node.work_unit.material_reqs = [MaterialReq('mat1', 50, 'stone')]
    ss = SimpleSynthetic(rand=random.Random(231))
    # wg = graph_restructuring(wg)
    landscape = ss.simple_synthetic_landscape(wg)
    scheduler = HEFTBetweenScheduler(work_estimator=work_time_estimator)
    # scheduler = GeneticScheduler(number_of_generation=10,
    #                              mutate_order=0.05,
    #                              mutate_resources=0.005,
    #                              size_of_population=10,
    #                              work_estimator=work_time_estimator)

    # Get information about created WorkGraph's attributes
    works_count = len(wg.nodes)
    work_names_count = len(set(n.work_unit.name for n in wg.nodes))
    res_kind_count = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
    print(works_count, work_names_count, res_kind_count)

    project = SchedulingPipeline.create() \
        .wg(wg) \
        .contractors(contractors) \
        .landscape(landscape) \
        .work_estimator(work_time_estimator) \
        .schedule(scheduler) \
        .visualization('2023-01-01') \
        .show_gant_chart()
