from sampo.userinput.parser.general_build import Graph
from typing import Iterable


def detect_cycles(input_graph: Iterable[object]) -> list | None:
    """
    Return list of cycles.

    :param input_graph: graph in abstract iterative form
    :return:
    """

    graph = Graph()

    for edge in input_graph:
        edge_info = tuple(edge)
        graph.add_edge(edge_info[0], edge_info[1])
    cycles = graph.eliminate_cycles(False)

    return cycles
