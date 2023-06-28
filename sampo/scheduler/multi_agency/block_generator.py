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
    Sequential = 0,
    Parallel = 1,
    Random = 2,
    Queues = 3


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
    :param obstruction_getter:
    :return: generated block graph
    """
    ss = SimpleSynthetic(rand)

    modes = rand.sample(list(SyntheticGraphType), counts=[p * n_blocks for p in type_prop], k=n_blocks)

    def generate_wg(mode, i):
        bottom_border, top_border = count_supplier(i)
        if bottom_border is not None:
            return ss.work_graph(mode, bottom_border=bottom_border)
        elif top_border is not None:
            return ss.work_graph(mode, top_border=top_border)
        else:
            return ss.work_graph(mode)

    nodes = [generate_wg(mode, i) for i, mode in enumerate(modes)]
    nodes += [generate_empty_graph(), generate_empty_graph()]
    bg = BlockGraph.pure(nodes, obstruction_getter)

    global_start, global_end = bg.nodes[-2:]

    match graph_type:
        case SyntheticBlockGraphType.Sequential:
            for idx, start in enumerate(bg.nodes[:-2]):
                if start.wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT \
                        and bg.nodes[idx + 1].wg.vertex_count > EMPTY_GRAPH_VERTEX_COUNT:
                    bg.add_edge(start, bg.nodes[idx + 1])
        case SyntheticBlockGraphType.Parallel:
            pass
        case SyntheticBlockGraphType.Random:
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
                         count_supplier: Callable[[int], tuple[int | None, int | None]],
                         edge_prob: float, rand: Random | None = Random(),
                         obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                         queues_num: int | None = None,
                         queues_blocks: list[int] | None = None,
                         queues_edges: list[int] | None = None,
                         logger: Callable[[str], None] = print) -> BlockGraph:
    if graph_type == SyntheticBlockGraphType.Queues:
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
    :param queues_num:
    :param queues_blocks:
    :param queues_edges:
    :return: generated block graph
    """
    ss = SimpleSynthetic(rand)
    nodes_all: list[BlockNode] = []
    parent: BlockNode = BlockNode(generate_empty_graph())

    for queue in range(queues_num):
        next_parent = BlockNode(generate_empty_graph())
        # generate vertices
        n_blocks = queues_blocks[queue]
        modes = rand.sample(list(SyntheticGraphType), counts=[p * n_blocks for p in type_prop], k=n_blocks)

        def generate_wg(mode, i):
            bottom_border, top_border = count_supplier(i)
            if bottom_border is not None:
                return ss.work_graph(mode, bottom_border=bottom_border)
            elif top_border is not None:
                return ss.work_graph(mode, top_border=top_border)
            else:
                return ss.work_graph(mode)
        nodes = [BlockNode(generate_wg(mode, i), obstruction_getter(i))
                 for i, mode in enumerate(modes)]
        nodes_all.append(parent)
        nodes_all.extend(nodes)

        # generate edges
        for i, node in enumerate(nodes):
            BlockGraph.add_edge(parent, node)
            BlockGraph.add_edge(node, next_parent)

        parent = next_parent

    nodes_all.append(parent)

    # logger(f'Generated queue {queue}: blocks={n_blocks}, edges={generated_edges}')
    logger('Queues')

    return BlockGraph(nodes_all)


