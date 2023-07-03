from enum import Enum
from random import Random
from typing import Callable
from uuid import uuid4

from sampo.generator import SimpleSynthetic
from sampo.generator.pipeline.types import SyntheticGraphType
from sampo.scheduler.multi_agency.block_graph import BlockGraph, BlockNode
from sampo.scheduler.utils.obstruction import Obstruction
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.works import WorkUnit


class SyntheticBlockGraphType(Enum):
    """
    Describe types of synthetic block graph:
    - Parallel - works can be performed mostly in parallel,
    - Sequential - works can be performed mostly in sequential,
    - Random - random structure of block graph,
    - Queue - queue structure typical of real processes capital construction.
    """
    SEQUENTIAL = 0
    PARALLEL = 1
    RANDOM = 2
    QUEUES = 3


EMPTY_GRAPH_VERTEX_COUNT = 2


def generate_empty_graph() -> WorkGraph:
    start = GraphNode(WorkUnit(str(uuid4()), ""), [])
    end = GraphNode(WorkUnit(str(uuid4()), ""), [start])
    return WorkGraph(start, end)


def generate_blocks(graph_type: SyntheticBlockGraphType, n_blocks: int, type_prop: list[int],
                    count_supplier: Callable[[int], tuple[int, int]],
                    edge_prob: float, rand: Random | None = Random(),
                    obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                    logger: Callable[[str], None] = print) -> BlockGraph:
    """
    Generate synthetic block graph according to given parameters.

    :param graph_type: type of BlockGraph
    :param n_blocks: the count of blocks
    :param type_prop: proportions of the `WorkGraph` types: General, Parallel, Sequential
    :param count_supplier: function that computes the borders of block size from it's index
    :param edge_prob: edge existence probability
    :param rand: a random reference
    :param obstruction_getter: obstruction, that can be inserted in BlockGraph
    :return: generated block graph
    """
    ss = SimpleSynthetic(rand)

    modes = rand.sample(list(SyntheticGraphType), counts=[p * n_blocks for p in type_prop], k=n_blocks)
    nodes = [ss.work_graph(mode, *count_supplier(i)) for i, mode in enumerate(modes)]
    nodes += [generate_empty_graph(), generate_empty_graph()]
    bg = BlockGraph.pure(nodes, obstruction_getter)

    global_start, global_end = bg.nodes[-2:]

    match graph_type:
        case SyntheticBlockGraphType.SEQUENTIAL:
            for idx, start in enumerate(bg.nodes[:-2]):
                if start.wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT \
                        and bg.nodes[idx + 1].wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT:
                    bg.add_edge(start, bg.nodes[idx + 1])
        case SyntheticBlockGraphType.PARALLEL:
            pass
        case SyntheticBlockGraphType.RANDOM:
            rev_edge_prob = int(1 / edge_prob)
            for idx, start in enumerate(bg.nodes):
                for end in bg.nodes[idx:]:
                    if start.wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT \
                            and end.wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT:
                        if start == end or rand.randint(0, rev_edge_prob) != 0:
                            continue
                        bg.add_edge(start, end)

    for node in bg.nodes:
        if node.wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT:
            bg.add_edge(global_start, node)
            bg.add_edge(node, global_end)

    logger(f'{graph_type.name} ' + ' '.join([str(mode.name) for mode in modes]))
    return bg


def generate_block_graph(graph_type: SyntheticBlockGraphType, n_blocks: int, type_prop: list[int],
                         count_supplier: Callable[[int], tuple[int, int]],
                         edge_prob: float, rand: Random | None = Random(),
                         obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                         queues_num: int | None = None,
                         queues_blocks: list[int] | None = None,
                         queues_edges: list[int] | None = None,
                         logger: Callable[[str], None] = print) -> BlockGraph:
    """
    Generate synthetic block graph of the received type.

    :param graph_type: type of Block Graph
    :param n_blocks: number of blocks
    :param type_prop: proportions of the `WorkGraph` types: General, Parallel, Sequential, Queues
    :param count_supplier: function that computes the borders of block size from it's index
    :param edge_prob: edge existence probability
    :param rand: a random reference
    :param obstruction_getter:
    :param queues_num: number of queues in a block graph
    :param queues_blocks: list of queues. It contains the number of blocks in each queue
    :param queues_edges:
    :param logger: for logging
    :return: generated block graph
    """
    if graph_type == SyntheticBlockGraphType.QUEUES:
        return generate_queues(type_prop, count_supplier, rand, obstruction_getter, queues_num, queues_blocks,
                               queues_edges, logger)
    else:
        return generate_blocks(graph_type, n_blocks, type_prop, count_supplier, edge_prob, rand, obstruction_getter,
                               logger)


def generate_queues(type_prop: list[int],
                    count_supplier: Callable[[int], tuple[int, int]],
                    rand: Random | None = Random(),
                    obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                    queues_num: int | None = None,
                    queues_blocks: list[int] | None = None,
                    queues_edges: list[int] | None = None,
                    logger: Callable[[str], None] = print) -> BlockGraph:
    """
    Generate synthetic block queues graph according to given parameters.

    :param type_prop: proportions of the `WorkGraph` types: General, Parallel, Sequential
    :param count_supplier: function that computes the borders of block size from it's index
    :param rand: a random reference
    :param obstruction_getter:
    :param queues_num: number of queues in a block graph
    :param queues_blocks: list of queues. It contains the number of blocks in each queue
    :param queues_edges:
    :return: generated block graph
    """
    ss = SimpleSynthetic(rand)
    nodes_all: list[BlockNode] = []
    nodes_prev: list[BlockNode] = []

    for queue in range(queues_num):
        # generate vertices
        n_blocks = queues_blocks[queue]
        modes = rand.sample(list(SyntheticGraphType), counts=[p * n_blocks for p in type_prop], k=n_blocks)
        nodes = [BlockNode(ss.work_graph(mode, *count_supplier(i)), obstruction_getter(i))
                 for i, mode in enumerate(modes)]
        nodes_all.extend(nodes)
        if not nodes_all:
            nodes_prev = nodes
            # logger(f'Generated queue 0: blocks={n_blocks}')
            continue

        # generate edges
        generated_edges = 0
        for i, node in enumerate(nodes[:-2]):
            if i >= len(nodes_prev):
                break
            BlockGraph.add_edge(node, nodes_prev[i])
            generated_edges += 1

        for i, node in enumerate(nodes):
            if i >= len(nodes_prev):
                break
            # we are going in reverse to fill edges that are not covered by previous cycle
            BlockGraph.add_edge(node, nodes_prev[-i])
            generated_edges += 1

        nodes_prev = nodes

        # logger(f'Generated queue {queue}: blocks={n_blocks}, edges={generated_edges}')
    logger('Queues')

    return BlockGraph(nodes_all)
