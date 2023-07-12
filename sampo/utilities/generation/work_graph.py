import pickle
from random import Random
from typing import Optional

from sampo.generator.environment.contractor import get_contractor, get_contractor_with_equal_proportions
from sampo.generator.pipeline.project import get_graph, SyntheticGraphType
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph


# TODO check outer usage
def graph_from_file(filepath: str, number_of_workers_in_contractors: int) -> tuple[WorkGraph, list[Contractor]]:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    wg: WorkGraph = data['work_graph']
    contractors: list[Contractor] = get_contractor_with_equal_proportions(number_of_workers_in_contractors)

    return wg, contractors


# TODO check outer usage
# Functions for generating synthetic graphs
def generate_work_graph(graph_mode: SyntheticGraphType, bottom_border: int,
                        rand: Optional[Random] = None) -> WorkGraph:
    return get_graph(mode=graph_mode,
                     bottom_border=int(bottom_border),
                     addition_cluster_probability=0,
                     rand=rand)


def generate_resources_pool(contractor_capacity: int, num_contractors: int = 1) -> list[Contractor]:
    return [get_contractor(pack_size, index=i) for i, pack_size in enumerate([contractor_capacity] * num_contractors)]
