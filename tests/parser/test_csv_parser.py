from sampo.parser.csv_parser import CSVParser
from sampo.parser.exception import WorkGraphBuildingException


def test_work_graph_csv_parser():
    try:
        wg = CSVParser.from_csv('parser/test_work_graph.csv')
        print(f'\n\nWork graph has works: {wg.nodes}\n\n')
    except Exception as e:
        raise WorkGraphBuildingException('There is no way to build work graph')