from sampo.userinput.parser.csv_parser import CSVParser
from sampo.userinput.parser.exception import WorkGraphBuildingException


def test_work_graph_csv_parser():
    try:
        works_info = CSVParser.read_graph_info(project_info='./test_work_info.csv',
                                               history_data='./test_history.csv')
        works_info.to_csv('repaired_connections.csv', sep=';')
        # wg, contractors = CSVParser.work_graph_and_contractors(works_info)
        # print(f'\n\nWork graph has works: {wg.nodes}, and the number of contractors is {len(contractors)}\n\n')
    except Exception as e:
        raise WorkGraphBuildingException(f'There is no way to build work graph, {e}')


