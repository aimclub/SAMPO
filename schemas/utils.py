import queue
import uuid
from random import Random
from typing import Optional
from uuid import uuid4

import networkx as nx

from schemas.works import AdjacencyMatrix
from schemas.graph import GraphNode, WorkGraph


def build_adjacency_graph(wg: WorkGraph) -> AdjacencyMatrix:
    """
    Build adjacency_graph and requirements dict from WorkGraph, check the graph for acyclic
    :param wg: work_graph
    :return: dict with adjacency graph and dict with requirements resources with each job
    """
    adjacency_graph = {}
    dict_for_building = []

    for node in wg.nodes:
        adjacency_node = set()
        for children in node.children:
            adjacency_node.add(children.id)
            dict_for_building.append((node, children.id))
        adjacency_graph[node.id] = adjacency_node

    g = nx.DiGraph(dict_for_building)
    assert (nx.is_directed_acyclic_graph(g))  # assert checking the graph for acyclic
    return adjacency_graph


def uuid_str(rand: Optional[Random] = None) -> str:
    ans = uuid4() if rand is None else uuid.UUID(int=rand.getrandbits(128))
    return str(ans)


def count_node_ancestors(finish: GraphNode, root: GraphNode) -> int:
    q = queue.Queue()
    count = 0
    used = set()
    used.add(root)
    q.put(finish)
    while not q.empty():
        node = q.get()
        for parent in node.parent_nodes:
            if parent in used:
                continue
            used.add(parent)
            q.put(parent)
            count += 1

    return count


def binary_search(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search_reversed(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] > x:
            lo = mid + 1
        else:
            hi = mid
    return lo
