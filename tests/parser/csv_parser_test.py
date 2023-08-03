from sampo.userinput.parser.csv_parser import CSVParser
from sampo.userinput.parser.exception import WorkGraphBuildingException


def test_work_graph_csv_parser():
    try:
        wg, contractors = CSVParser.work_graph_and_contractors('tests/parser/test_work_graph.csv',
                                                               contractor_info='tests/parser/test_contractors.csv')
        print(f'\n\nWork graph has works: {wg.nodes}, and the number of contractors is {len(contractors)}\n\n')
    except Exception as e:
        raise WorkGraphBuildingException(f'There is no way to build work graph, {e}')
