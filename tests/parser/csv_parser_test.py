import os
import sys

import pandas as pd

from sampo.userinput.parser.csv_parser import CSVParser
from sampo.userinput.parser.exception import WorkGraphBuildingException


def test_work_graph_csv_parser():
    try:
        history = pd.DataFrame(columns=['marker_for_glue', 'work_name', 'first_day', 'last_day',
                                                            'upper_works', 'work_name_clear_old', 'smr_name',
                                                            'work_name_clear', 'granular_smr_name'])
        works_info = CSVParser.read_graph_info(project_info=os.path.join(sys.path[0], 'tests/parser/test_wg_no_connections.csv'),
                                               history_data=history,
                                               full_connections=True)
        works_info.to_csv(os.path.join(sys.path[0], 'tests/parser/repaired.csv'), sep=';')
        wg, contractors = CSVParser.work_graph_and_contractors(works_info)
        print(f'\n\nWork graph has works: {wg.nodes}, and the number of contractors is {len(contractors)}\n\n')
    except Exception as e:
        raise WorkGraphBuildingException(f'There is no way to build work graph, {e}')

    os.remove(os.path.join(sys.path[0], 'tests/parser/repaired.csv'))
