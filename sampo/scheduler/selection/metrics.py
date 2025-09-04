"""Metrics and helper functions for graph-based scheduling.

Метрики и вспомогательные функции для графового планирования.
"""

from collections import defaultdict
from math import ceil

import numpy as np
import torch

from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.graph import WorkGraph


def one_hot_encode(v, max_v):
    """Convert index into one-hot vector.

    Преобразует индекс в one-hot вектор.

    Args:
        v: Index to encode.
            Индекс для кодирования.
        max_v: Length of result vector.
            Длина результирующего вектора.

    Returns:
        One-hot encoded list.
        Список в формате one-hot.
    """
    res = [float(0) for _ in range(max_v)]
    res[v] = float(1)
    return res


def one_hot_decode(v: torch.Tensor):
    """Decode one-hot tensor back to index.

    Декодирует one-hot тензор обратно в индекс.

    Args:
        v: Tensor to decode.
            Тензор для декодирования.

    Returns:
        Decoded index.
        Декодированный индекс.
    """
    for i in range(len(v)):
        if v[i] == 1:
            return i


def metric_resource_constrainedness(wg: WorkGraph) -> list[float]:
    """Calculate constrainedness for each resource type.

    Вычисляет степень ограниченности для каждого типа ресурса.

    The constrainedness equals the average requested units divided by the
    capacity of the resource.

    Ограниченность равна среднему количеству запрошенных единиц, делённому
    на вместимость ресурса.

    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.

    Returns:
        List of constrainedness coefficients.
        Список коэффициентов ограниченности.
    """
    resource_dict = defaultdict(lambda: [0, 0])

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind][0] += 1
            resource_dict[req.kind][1] += req.volume

    return [value[0] / value[1] for name, value in resource_dict.items()]


def metric_graph_parallelism_degree(wg: WorkGraph) -> list[float]:
    """Estimate degree of parallel execution per graph level.

    Оценивает степень параллельного выполнения по уровням графа.

    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.

    Returns:
        Averaged parallelism degrees for batches.
        Усреднённые степени параллельности по батчам.
    """
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
    """Compute length of the longest path in graph.

    Вычисляет длину самого длинного пути в графе.

    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.

    Returns:
        Length of the longest path.
        Длина самого длинного пути.
    """
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
    """Return number of vertices in graph.

    Возвращает число вершин в графе.

    Args:
        wg: Work graph.
            Граф работ.

    Returns:
        Vertex count.
        Количество вершин.
    """
    return wg.vertex_count


def metric_average_work_per_activity(wg: WorkGraph) -> float:
    """Average work volume per node.

    Средний объём работы на узел.

    Args:
        wg: Work graph.
            Граф работ.

    Returns:
        Average work volume.
        Средний объём работы.
    """
    return sum(node.work_unit.volume for node in wg.nodes) / wg.vertex_count


def metric_relative_max_children(wg: WorkGraph) -> float:
    """Relative maximum number of children for a node.

    Относительное максимальное число потомков для узла.

    Args:
        wg: Work graph.
            Граф работ.

    Returns:
        Ratio of maximum children to total vertices.
        Отношение максимального числа потомков к числу вершин.
    """
    return max((len(node.children) for node in wg.nodes if node.children)) / wg.vertex_count


def metric_average_resource_usage(wg: WorkGraph) -> float:
    """Average number of requested workers per node.

    Среднее число требуемых работников на узел.

    Args:
        wg: Work graph.
            Граф работ.

    Returns:
        Average resource usage.
        Среднее использование ресурсов.
    """
    return sum(sum((req.min_count + req.max_count) / 2 for req in node.work_unit.worker_reqs)
               for node in wg.nodes) / wg.vertex_count


def metric_relative_max_parents(wg: WorkGraph) -> float:
    """Relative maximum number of parents for a node.

    Относительное максимальное число родителей для узла.

    Args:
        wg: Work graph.
            Граф работ.

    Returns:
        Ratio of maximum parents to total vertices.
        Отношение максимального числа родителей к числу вершин.
    """
    return max((len(node.parents) for node in wg.nodes if node.parents)) / wg.vertex_count


def encode_graph(wg: WorkGraph) -> list[float]:
    """Encode graph structure into feature vector.

    Кодирует структуру графа в вектор признаков.

    Args:
        wg: Work graph to encode.
            Граф работ для кодирования.

    Returns:
        List of graph features.
        Список признаков графа.
    """
    return [
        metric_vertex_count(wg),
        metric_average_work_per_activity(wg),
        metric_relative_max_children(wg),
        metric_average_resource_usage(wg),
        metric_longest_path(wg),
        *metric_graph_parallelism_degree(wg)
    ]

