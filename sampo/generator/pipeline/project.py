from random import Random
from typing import Callable

from sampo.generator.config.gen_counts import MIN_GRAPH_COUNTS, ADDITION_CLUSTER_PROBABILITY, GRAPH_COUNTS, \
    MAX_BOREHOLES_PER_BLOCK, BRANCHING_PROBABILITY
from sampo.generator.pipeline.cluster import get_start_stage, get_cluster_works, get_finish_stage, add_addition_work
from sampo.generator.types import SyntheticGraphType, StageType
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.utils import count_node_ancestors


def get_small_graph(cluster_name: str | None = 'C1', rand: Random | None = None) -> WorkGraph:
    """
    Creates a small graph of works consisting of 30-50 vertices;
    :param cluster_name: str - the first cluster name
    :param rand: Optional[Random] - generator of numbers with a given seed or None
    :return:
       work_graph: WorkGraph - work graph where count of vertex between 30 and 50
    """

    s = get_start_stage()
    pipe_nodes_count = MIN_GRAPH_COUNTS['pipe_nodes'].rand_int(rand)
    pipe_net_count = MIN_GRAPH_COUNTS['pipe_net'].rand_int(rand)
    light_masts_count = MIN_GRAPH_COUNTS['light_masts'].rand_int(rand)
    borehole_count = MIN_GRAPH_COUNTS['borehole'].rand_int(rand)

    c1, _ = get_cluster_works(root_node=s, cluster_name=cluster_name, pipe_nodes_count=pipe_nodes_count,
                              pipe_net_count=pipe_net_count, light_masts_count=light_masts_count,
                              borehole_counts=[borehole_count], rand=rand)
    f = get_finish_stage([c1])
    graph = WorkGraph(s, f)
    return graph


def get_cluster_graph(root_node: GraphNode, cluster_name: str, pipe_nodes_count: int | None = None,
                      pipe_net_count: int | None = None, light_masts_count: int | None = None,
                      borehole_counts: list[int] | None = None, add_addition_cluster: bool | None = False,
                      addition_cluster_probability: float | None = ADDITION_CLUSTER_PROBABILITY,
                      rand: Random | None = None) -> (list[GraphNode], dict[str, GraphNode]):
    pipe_nodes_count = pipe_nodes_count or GRAPH_COUNTS['pipe_nodes'].rand_int(rand)
    pipe_net_count = pipe_net_count or GRAPH_COUNTS['pipe_net'].rand_int(rand)
    light_masts_count = light_masts_count or GRAPH_COUNTS['light_masts'].rand_int(rand)
    if borehole_counts is None:
        counts = GRAPH_COUNTS['borehole'].rand_int(rand)
        if counts <= MAX_BOREHOLES_PER_BLOCK:
            borehole_counts = [counts]
        else:
            borehole_counts = [MAX_BOREHOLES_PER_BLOCK, counts - MAX_BOREHOLES_PER_BLOCK]

    c_master, roads = get_cluster_works(root_node=root_node, cluster_name=cluster_name,
                                        pipe_nodes_count=pipe_nodes_count, pipe_net_count=pipe_net_count,
                                        light_masts_count=light_masts_count, borehole_counts=borehole_counts, rand=rand)

    checkpoints = [c_master]
    if add_addition_cluster or add_addition_work(addition_cluster_probability, rand):
        c_slave, _ = get_cluster_works(root_node=root_node, cluster_name=cluster_name,
                                       pipe_nodes_count=pipe_nodes_count, pipe_net_count=pipe_net_count,
                                       light_masts_count=light_masts_count, borehole_counts=borehole_counts,
                                       roads=roads, rand=rand)
        checkpoints.append(c_slave)
    return checkpoints, roads


def get_graph(mode: SyntheticGraphType | None = SyntheticGraphType.General,
              cluster_name_prefix: str | None = 'C',
              cluster_counts: int | None = 0,
              branching_probability: float | None = BRANCHING_PROBABILITY,
              addition_cluster_probability: float | None = ADDITION_CLUSTER_PROBABILITY,
              bottom_border: int | None = 0,
              top_border: int | None = 0,
              rand: Random | None = None) -> WorkGraph:
    """
    Invokes a graph of the given type if at least one positive value of
        cluster_counts, addition_cluster_probability, bottom_border is given;
    :param mode: str - 'general' or 'sequence' or 'parallel - the type of the returned graph
    :param cluster_name_prefix: str -  cluster name prefix, if the prefix is 'C',
        the clusters will be called 'C1', 'C2' etc.
    :param cluster_counts: Optional[int] - Number of clusters for the graph
    :param branching_probability: Optional[float] - The probability that the node will not be connected to the last
        cluster, but to any other cluster for the general mode
    :param addition_cluster_probability: Optional[float] - probability of a slave (example C3_1) cluster
        to the main cluster
    :param bottom_border: Optional[int] - bottom border for number of works for the graph
    :param top_border: Optional[int] - top border for number of works for the graph
    :param rand: Optional[Random] - generator of numbers with a given seed or None
    :return:
        work_graph: WorkGraph - the desired work graph
    """
    assert cluster_counts + bottom_border + top_border > 0, 'At least one border param should be specified'
    assert cluster_counts >= 0 and branching_probability >= 0 and top_border >= 0, 'Params should not be negative'

    rand = rand or Random()
    get_root_stage: Callable[[list[StageType], float, Random], GraphNode] = graph_mode_to_callable(mode)

    if bottom_border > 0:
        top_border = 0
    masters_clusters_ind = 1
    works_generated = 0
    s = get_start_stage()
    stages: list[StageType] = [(s, {'min': s})]

    while True:
        if cluster_counts > 0 and (cluster_counts == len(stages) - 1):
            addition_cluster_probability = 0

        root_stage = get_root_stage(stages, branching_probability, rand)
        checkpoints, r = get_cluster_graph(root_stage, f'{cluster_name_prefix}{masters_clusters_ind}',
                                           addition_cluster_probability=addition_cluster_probability, rand=rand)
        tmp_finish = get_finish_stage(checkpoints)
        count_works = count_node_ancestors(tmp_finish, root_stage)

        if 0 < top_border < (count_works + works_generated):
            break

        stages += [(c, r) for c in checkpoints]
        masters_clusters_ind += 1
        works_generated += count_works

        if 0 < bottom_border <= works_generated or 0 < cluster_counts <= (len(stages) - 1):
            break

    if len(stages) == 1:
        return get_small_graph(cluster_name=f'{cluster_name_prefix}1')
    f = get_finish_stage([c for c, _ in stages[1:]])
    graph = WorkGraph(s, f)
    return graph


def graph_mode_to_callable(mode: SyntheticGraphType) -> \
        Callable[[list[tuple[GraphNode, dict[str, GraphNode]]], float, Random], GraphNode]:
    if mode is SyntheticGraphType.General:
        return general_graph_mode
    if mode is SyntheticGraphType.Parallel:
        return parallel_graph_mode_get_root
    return sequence_graph_mode_get_root


def parallel_graph_mode_get_root(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                                 branching_probability: float, rand: Random) -> GraphNode:
    return stages[0][0]


def sequence_graph_mode_get_root(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                                 branching_probability: float, rand: Random) -> GraphNode:
    return stages[-1][0]


def general_graph_mode(stages: list[tuple[GraphNode, dict[str, GraphNode]]],
                       branching_probability: float, rand: Random) -> GraphNode:
    is_branching = rand.random() <= branching_probability
    ind = len(stages) - 1
    if is_branching and len(stages) > 2:
        ind = rand.randint(1, len(stages) - 2)
    return stages[ind][1]['min']
