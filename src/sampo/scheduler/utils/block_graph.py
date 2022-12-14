from copy import deepcopy
from operator import attrgetter
from random import Random
from typing import Callable

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.utilities.collections import build_index


class BlockNode:
    """
    `BlockNode` represents the node of `BlockGraph` and contains corresponding
    `WorkGraph` and related dependencies between blocks in `BlockGraph`.
    """
    def __init__(self, wg: WorkGraph):
        self.wg = wg
        self.blocks_from: list[BlockNode] = []
        self.blocks_to: list[BlockNode] = []

    @property
    def id(self):
        return self.wg.start.id


class BlockGraph:
    """
    Represents the block graph, where blocks are instances of `WorkGraph` and
    edges are simple *FS* dependencies.
    """
    def __init__(self, nodes: list[WorkGraph]):
        self.nodes = [BlockNode(node) for node in nodes]
        self.node_dict = build_index(self.nodes, attrgetter('id'))

    def __getitem__(self, item) -> BlockNode:
        return self.node_dict[item]

    def to_work_graph(self) -> WorkGraph:
        """
        Creates `WorkGraph` that are equal to this `BlockGraph`.
        """
        copied_nodes = deepcopy(self.nodes)
        global_start: GraphNode = [node for node in copied_nodes if len(node.blocks_from) == 0][0].wg.start
        global_end:   GraphNode = [node for node in copied_nodes if len(node.blocks_to) == 0][0].wg.finish

        for end in copied_nodes:
            end.wg.start.add_parents([start.wg.finish for start in end.blocks_from])

        return WorkGraph(global_start, global_end)

    @staticmethod
    def add_edge(start: BlockNode, end: BlockNode):
        start.blocks_to.append(end)
        end.blocks_from.append(start)


def generate_blocks(n_blocks: int, type_prop: list[int],
                    count_supplier: Callable[[int], tuple[int, int]],
                    edge_prob: float, rand: Random | None = Random()) -> BlockGraph:
    """
    Generate block graph according to given parameters.

    :param n_blocks: the count of blocks
    :param type_prop: proportions of the `WorkGraph` types: General, Parallel, Sequential
    :param count_supplier: function that computes the borders of block size from it's index
    :param edge_prob: edge existence probability
    :param rand: a random reference
    :return: generated block graph
    """
    ss = SimpleSynthetic(rand)

    modes = rand.sample(list(SyntheticGraphType), counts=[p * n_blocks for p in type_prop], k=n_blocks)
    nodes = [ss.work_graph(mode, *count_supplier(i)) for i, mode in enumerate(modes)]
    bg = BlockGraph(nodes)

    rev_edge_prob = int(1 / edge_prob)
    for idx, start in enumerate(bg.nodes):
        for end in bg.nodes[idx:]:
            if start == end or rand.randint(0, rev_edge_prob) != 0:
                continue
            bg.add_edge(start, end)

    return bg
