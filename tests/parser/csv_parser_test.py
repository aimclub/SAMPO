import os
import sys
from operator import attrgetter

import pandas as pd

from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import HEFTScheduler
from sampo.userinput.parser.csv_parser import CSVParser
from sampo.userinput.parser.exception import WorkGraphBuildingException


def test_work_graph_csv_parser_without_history():
    history = pd.DataFrame(columns=['marker_for_glue', 'model_name', 'first_day', 'last_day',
                                    'upper_works', 'work_name_clear_old', 'smr_name',
                                    'work_name_clear', 'granular_smr_name'])
    works_info = CSVParser.read_graph_info(project_info=os.path.join(sys.path[0], 'tests/parser/test_wg.csv'),
                                           history_data=history,
                                           all_connections=True)
    wg, contractors = CSVParser.work_graph_and_contractors(works_info)
    print(f'\n\nWork graph has works: {wg.nodes}, and the number of contractors is {len(contractors)}\n\n')


def test_work_graph_csv_parser_with_history():
    works_info = CSVParser.read_graph_info(project_info=os.path.join(sys.path[0], 'tests/parser/test_wg.csv'),
                                           history_data=os.path.join(sys.path[0], 'tests/parser/test_history_data.csv'),
                                           all_connections=False,
                                           change_connections_info=True)
    wg, contractors = CSVParser.work_graph_and_contractors(works_info)
    print(f'\n\nWork graph has works: {wg.nodes}, and the number of contractors is {len(contractors)}\n\n')


def test_work_graph_frame_serialization(setup_wg):
    frame = setup_wg.to_frame()

    rebuilt_wg = CSVParser.work_graph(frame)

    project = SchedulingPipeline.create().wg(rebuilt_wg).schedule(HEFTScheduler()).finish()[0]

    origin_ids = set([node.id for node in setup_wg.nodes if not node.work_unit.is_service_unit])
    rebuilt_ids = set([node.id for node in rebuilt_wg.nodes if not node.work_unit.is_service_unit])

    assert origin_ids == rebuilt_ids
