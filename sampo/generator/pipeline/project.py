"""Utilities for generating synthetic project graphs.

Утилиты для генерации синтетических графов проекта.
"""

from random import Random
from typing import Callable

from sampo.generator.config.gen_counts import (
    ADDITION_CLUSTER_PROBABILITY,
    BRANCHING_PROBABILITY,
    GRAPH_COUNTS,
    MAX_BOREHOLES_PER_BLOCK,
    MIN_GRAPH_COUNTS,
)
from sampo.generator.pipeline import StageType, SyntheticGraphType
from sampo.generator.pipeline.cluster import _add_addition_work, get_cluster_works
from sampo.schemas.graph import GraphNode, WorkGraph


def get_small_graph(cluster_name: str | None = 'C1', rand: Random | None = None) -> WorkGraph:
    """Create a small work graph with 30-50 vertices.

    Создаёт небольшой граф работ, содержащий 30–50 вершин.

    Args:
        cluster_name (str | None): Name of the initial cluster.
            Имя первого кластера.
        rand (Random | None): Random number generator.
            Генератор случайных чисел.

    Returns:
        WorkGraph: Work graph containing between 30 and 50 vertices.
        WorkGraph: Граф работ, включающий от 30 до 50 вершин.
    """

    pipe_nodes_count = MIN_GRAPH_COUNTS['pipe_nodes'].rand_int(rand)
    pipe_net_count = MIN_GRAPH_COUNTS['pipe_net'].rand_int(rand)
    light_masts_count = MIN_GRAPH_COUNTS['light_masts'].rand_int(rand)
    borehole_count = MIN_GRAPH_COUNTS['borehole'].rand_int(rand)

    c1, c_nodes, _ = get_cluster_works(cluster_name=cluster_name, pipe_nodes_count=pipe_nodes_count,
                                       pipe_net_count=pipe_net_count, light_masts_count=light_masts_count,
                                       borehole_counts=[borehole_count], rand=rand)
    return WorkGraph.from_nodes(list(c_nodes.values()) + [c1])


def _get_cluster_graph(cluster_name: str, pipe_nodes_count: int | None = None,
                       pipe_net_count: int | None = None, light_masts_count: int | None = None,
                       borehole_counts: list[int] | None = None, add_addition_cluster: bool | None = False,
                       addition_cluster_probability: float | None = ADDITION_CLUSTER_PROBABILITY,
                       rand: Random | None = None) -> tuple[list[GraphNode], dict[str, GraphNode], int]:
    pipe_nodes_count = pipe_nodes_count or GRAPH_COUNTS['pipe_nodes'].rand_int(rand)
    pipe_net_count = pipe_net_count or GRAPH_COUNTS['pipe_net'].rand_int(rand)
    light_masts_count = light_masts_count or GRAPH_COUNTS['light_masts'].rand_int(rand)
    if borehole_counts is None:
        counts = GRAPH_COUNTS['borehole'].rand_int(rand)
        if counts <= MAX_BOREHOLES_PER_BLOCK:
            borehole_counts = [counts]
        else:
            borehole_counts = [MAX_BOREHOLES_PER_BLOCK, counts - MAX_BOREHOLES_PER_BLOCK]

    # the whole count of nodes generated in this function
    count_nodes = 0
    c_master, roads, count_master = get_cluster_works(cluster_name=cluster_name,
                                                      pipe_nodes_count=pipe_nodes_count, pipe_net_count=pipe_net_count,
                                                      light_masts_count=light_masts_count,
                                                      borehole_counts=borehole_counts, rand=rand)
    count_nodes += count_master

    checkpoints = [c_master]
    if add_addition_cluster or _add_addition_work(addition_cluster_probability, rand):
        c_slave, _, count_slave = get_cluster_works(
            cluster_name=cluster_name,
            pipe_nodes_count=pipe_nodes_count,
            pipe_net_count=pipe_net_count,
            light_masts_count=light_masts_count,
            borehole_counts=borehole_counts,
            roads=roads,
            rand=rand,
        )
        count_nodes += count_slave
        checkpoints.append(c_slave)
    return checkpoints, roads, count_nodes


def get_graph(mode: SyntheticGraphType | None = SyntheticGraphType.GENERAL,
              cluster_name_prefix: str | None = 'C',
              cluster_counts: int | None = 0,
              branching_probability: float | None = BRANCHING_PROBABILITY,
              addition_cluster_probability: float | None = ADDITION_CLUSTER_PROBABILITY,
              bottom_border: int | None = 0,
              top_border: int | None = 0,
              rand: Random | None = None) -> WorkGraph:
    """Generate a synthetic work graph of the specified type.

    Генерирует синтетический граф работ указанного типа.

    Args:
        mode (SyntheticGraphType | None): Type of graph to generate.
            Тип генерируемого графа.
        cluster_name_prefix (str | None): Prefix used for cluster names.
            Префикс, используемый для имен кластеров.
        cluster_counts (int | None): Desired number of clusters in the graph.
            Требуемое число кластеров в графе.
        branching_probability (float | None): Probability of connecting a
            cluster to a non-sequential predecessor.
            Вероятность соединения кластера с неочередным предшественником.
        addition_cluster_probability (float | None): Probability of adding a
            slave cluster.
            Вероятность добавления подчинённого кластера.
        bottom_border (int | None): Minimum number of works in the graph.
            Минимальное количество работ в графе.
        top_border (int | None): Maximum number of works in the graph.
            Максимальное количество работ в графе.
        rand (Random | None): Random number generator.
            Генератор случайных чисел.

    Returns:
        WorkGraph: Generated work graph.
        WorkGraph: Сгенерированный граф работ.
    """
    assert cluster_counts + bottom_border + top_border > 0, 'At least one border param should be specified'
    assert cluster_counts >= 0 and branching_probability >= 0 and top_border >= 0, 'Params should not be negative'

    rand = rand or Random()
    get_root_stage: Callable[[list[StageType], float, Random], GraphNode] = _graph_mode_to_callable(mode)

    if bottom_border > 0:
        top_border = 0
    masters_clusters_ind = 1
    works_generated = 0
    stages = []

    while True:
        if cluster_counts > 0 and cluster_counts == len(stages) - 1:
            addition_cluster_probability = 0

        checkpoints, roads, count_works = _get_cluster_graph(f'{cluster_name_prefix}{masters_clusters_ind}',
                                                             addition_cluster_probability=addition_cluster_probability,
                                                             rand=rand)
        root_stage = get_root_stage(stages, branching_probability, rand)
        if root_stage is not None:
            for checkpoint in checkpoints:
                checkpoint.add_parents([root_stage])

        stages += [(c, roads) for c in checkpoints]
        masters_clusters_ind += 1
        works_generated += count_works

        if (0 < bottom_border <= works_generated or 0 < top_border < count_works + works_generated
                or 0 < cluster_counts <= (len(stages) - 1)):
            break

    if len(stages) == 1:
        return get_small_graph(cluster_name=f'{cluster_name_prefix}1')

    nodes = [road for _, roads in stages for road in roads.values()]
    nodes.extend([c for c, _ in stages])
    return WorkGraph.from_nodes(nodes)


def _graph_mode_to_callable(mode: SyntheticGraphType) -> \
        Callable[[list[tuple[GraphNode, dict[str, GraphNode]]], float, Random], GraphNode]:
    if mode is SyntheticGraphType.GENERAL:
        return _general_graph_mode
    if mode is SyntheticGraphType.PARALLEL:
        return _parallel_graph_mode_get_root
    return _sequence_graph_mode_get_root


def _parallel_graph_mode_get_root(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                                  branching_probability: float, rand: Random) -> GraphNode | None:
    if len(stages) == 0:
        return None
    return stages[0][0]


def _sequence_graph_mode_get_root(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                                  branching_probability: float, rand: Random) -> GraphNode | None:
    if len(stages) == 0:
        return None
    return stages[-1][0]


def _general_graph_mode(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                        branching_probability: float, rand: Random) -> GraphNode | None:
    is_branching = rand.random() <= branching_probability
    ind = len(stages) - 1
    if is_branching and len(stages) > 2:
        ind = rand.randint(1, len(stages) - 2)
    if len(stages) == 0:
        return None
    return stages[ind][1]['min']
