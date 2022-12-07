from random import Random
from typing import Callable

from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.schemas.graph import WorkGraph


class BlockNode:
    def __init__(self, wg: WorkGraph):
        self.wg = wg
        self.blocks_from: list[BlockNode] = []
        self.blocks_to: list[BlockNode] = []

    @property
    def id(self):
        return self.wg.start.id


class BlockGraph:
    def __init__(self, nodes: list[WorkGraph]):
        self.nodes = [BlockNode(node) for node in nodes]

    @staticmethod
    def add_edge(start: BlockNode, end: BlockNode):
        start.blocks_to.append(end)
        end.blocks_from.append(start)


def generate_blocks(n_blocks: int, type_prop: list[int],
                    count_supplier: Callable[[int], tuple[int, int]],
                    edge_prob: float, rand: Random | None = Random()) -> BlockGraph:
    """
    Generate block graph according to given parameters

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
