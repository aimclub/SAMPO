from collections import defaultdict
from math import ceil

import numpy as np
import torch

from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.graph import WorkGraph


def one_hot_encode(v, max_v):
    res = [float(0) for _ in range(max_v)]
    res[v] = float(1)
    return res


def one_hot_decode(v: torch.Tensor):
    for i in range(len(v)):
        if v[i] == 1:
            return i


def metric_resource_constrainedness(wg: WorkGraph) -> list[float]:
    """
    The resource constrainedness of a resource type k is defined as  the average number of units requested by all
    activities divided by the capacity of the resource type

    :param wg: Work graph
    :return: List of RC coefficients for each resource type
    """
    resource_dict = defaultdict(lambda: [0, 0])

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind][0] += 1
            resource_dict[req.kind][1] += req.volume

    return [value[0] / value[1] for name, value in resource_dict.items()]


def metric_graph_parallelism_degree(wg: WorkGraph) -> list[float]:
    batches = 8
    parallelism_degree = []
    current_node = wg.start
    node_count = wg.vertex_count

    stack = [current_node]
    while stack:
        tmp_stack = set()
        parallelism_coef = 0

        for i in range(len(stack)):
            if stack[i] == 0:
                continue
            for j in range(i + 1, len(stack)):
                if stack[j] == 0:
                    continue
                if stack[j] in stack[i].children:
                    stack[j] = 0

        for node in stack:
            if node == 0:
                continue
            parallelism_coef += 1
            for child in node.children:
                tmp_stack.add(child)
        parallelism_degree.append(parallelism_coef / node_count)
        stack = list(tmp_stack)

    step = ceil(len(parallelism_degree) / batches)
    aggregated_degree = [0] * batches
    for i in range(0, len(parallelism_degree), step):
        aggregated_degree[i // step] = np.mean(parallelism_degree[i:(i + step)])

    return aggregated_degree


def metric_longest_path(wg: WorkGraph) -> float:
    scheduler = TopologicalScheduler()
    stack = scheduler.prioritization(wg, None)

    dist = {}
    for node in stack:
        dist[node.id] = 0

    while stack:
        node = stack.pop()
        for child in node.children:
            dist[child.id] = max(dist[child.id], dist[node.id] + 1)

    return max(dist.values())


def metric_vertex_count(wg: WorkGraph) -> float:
    return wg.vertex_count


def metric_average_work_per_activity(wg: WorkGraph) -> float:
    return sum(node.work_unit.volume for node in wg.nodes) / wg.vertex_count


def metric_relative_max_children(wg: WorkGraph) -> float:
    return max((len(node.children) for node in wg.nodes if node.children)) / wg.vertex_count


def metric_average_resource_usage(wg: WorkGraph) -> float:
    return sum(sum((req.min_count + req.max_count) / 2 for req in node.work_unit.worker_reqs)
               for node in wg.nodes) / wg.vertex_count


def metric_relative_max_parents(wg: WorkGraph) -> float:
    return max((len(node.parents) for node in wg.nodes if node.parents)) / wg.vertex_count


def encode_graph(wg: WorkGraph) -> list[float]:
    return [
        metric_vertex_count(wg),
        metric_average_work_per_activity(wg),
        metric_relative_max_children(wg),
        metric_average_resource_usage(wg),
        metric_longest_path(wg),
        *metric_graph_parallelism_degree(wg)
    ]
