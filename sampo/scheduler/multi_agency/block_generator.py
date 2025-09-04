"""Utilities for generating synthetic block graphs for the multi-agent scheduler.

Инструменты для генерации синтетических графов блоков для многоагентного
планировщика.
"""

from enum import Enum
from random import Random
from typing import Callable
from uuid import uuid4

from sampo.generator.base import SimpleSynthetic
from sampo.generator.pipeline import SyntheticGraphType
from sampo.scheduler.multi_agency.block_graph import BlockGraph, BlockNode
from sampo.scheduler.utils.obstruction import Obstruction
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.works import WorkUnit


class SyntheticBlockGraphType(Enum):
    """Types of synthetic block graphs.

    Типы синтетических графов блоков.

    Attributes:
        SEQUENTIAL: Works are performed mostly sequentially.
            Работы выполняются преимущественно последовательно.
        PARALLEL: Works can be performed mostly in parallel.
            Работы могут выполняться преимущественно параллельно.
        RANDOM: Random structure of a block graph.
            Случайная структура графа блоков.
        QUEUES: Queue structure typical of real construction processes.
            Очередная структура, типичная для процессов капитального строительства.
    """
    SEQUENTIAL = 0
    PARALLEL = 1
    RANDOM = 2
    QUEUES = 3


EMPTY_GRAPH_VERTEX_COUNT = 2


def generate_empty_graph() -> WorkGraph:
    """Create a minimal work graph with start and end nodes.

    Создать минимальный граф работ с начальными и конечными вершинами.

    Returns:
        WorkGraph: Generated empty graph.
            Сгенерированный пустой граф.
    """

    start = GraphNode(WorkUnit(str(uuid4()), ""), [])
    end = GraphNode(WorkUnit(str(uuid4()), ""), [start])
    return WorkGraph(start, end)


def generate_blocks(graph_type: SyntheticBlockGraphType, n_blocks: int, type_prop: list[int],
                    count_supplier: Callable[[int], tuple[int, int]],
                    edge_prob: float, rand: Random | None = Random(),
                    obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                    logger: Callable[[str], None] = print) -> BlockGraph:
    """Generate a synthetic block graph.

    Сгенерировать синтетический граф блоков.

    Args:
        graph_type: Type of the resulting block graph.
            Тип результирующего графа блоков.
        n_blocks: Number of blocks.
            Количество блоков.
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный.
        count_supplier: Function that returns size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        edge_prob: Probability that an edge exists.
            Вероятность существования ребра.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.

    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
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
                    if not start.is_service() and not end.is_service():
                        if start == end or rand.randint(0, rev_edge_prob) != 0:
                            continue
                        bg.add_edge(start, end)

    for node in bg.nodes:
        if not node.is_service():
            bg.add_edge(global_start, node)
            bg.add_edge(node, global_end)

    logger(f'{graph_type.name} ' + ' '.join([str(mode.name) for i, mode in enumerate(modes)
                                             if nodes[i].vertex_count != EMPTY_GRAPH_VERTEX_COUNT]))
    return bg


def generate_block_graph(graph_type: SyntheticBlockGraphType, n_blocks: int, type_prop: list[int],
                         count_supplier: Callable[[int], tuple[int | None, int | None]],
                         edge_prob: float, rand: Random | None = Random(),
                         obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None,
                         queues_num: int | None = None,
                         queues_blocks: list[int] | None = None,
                         queues_edges: list[int] | None = None,
                         logger: Callable[[str], None] = print) -> BlockGraph:
    """Generate a block graph of the given type.

    Сгенерировать граф блоков заданного типа.

    Args:
        graph_type: Desired structure of the block graph.
            Требуемая структура графа блоков.
        n_blocks: Number of blocks.
            Количество блоков.
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential, Queues.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный, очереди.
        count_supplier: Function returning size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        edge_prob: Probability that an edge exists.
            Вероятность существования ребра.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        queues_num: Number of queues in the block graph.
            Количество очередей в графе блоков.
        queues_blocks: Number of blocks in each queue.
            Количество блоков в каждой очереди.
        queues_edges: Number of edges in each queue.
            Количество рёбер в каждой очереди.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.

    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
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
    """Generate a block graph with queue structure.

    Сгенерировать граф блоков с очередной структурой.

    Args:
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный.
        count_supplier: Function returning size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        queues_num: Number of queues in the block graph.
            Количество очередей в графе блоков.
        queues_blocks: Number of blocks in each queue.
            Количество блоков в каждой очереди.
        queues_edges: Number of edges in each queue.
            Количество рёбер в каждой очереди.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.

    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
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
    logger('QUEUES')

    return BlockGraph(nodes_all)
