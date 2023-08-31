from sampo.schemas.graph import WorkGraph


def one_hot_encode(v, max_v):
    res = [float(0) for _ in range(max_v)]
    res[v] = float(1)
    return res


def metric_resource_constrainedness(wg: WorkGraph) -> list[float]:
    """
    The resource constrainedness of a resource type k is defined as  the average number of units requested by all
    activities divided by the capacity of the resource type

    :param wg: Work graph
    :return: List of RC coefficients for each resource type
    """
    rc_coefs = []
    resource_dict = {}

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind] = {'activity_amount': 1, 'volume': 0}

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind]['activity_amount'] += 1
            resource_dict[req.kind]['volume'] += req.volume

    for name, value in resource_dict.items():
        rc_coefs.append(value['activity_amount'] / value['volume'])

    return rc_coefs


def metric_graph_parallelism_degree(wg: WorkGraph) -> list[float]:
    parallelism_degree = []
    current_node = wg.start

    stack = [current_node]
    while stack:
        tmp_stack = []
        parallelism_coef = 0
        for node in stack:
            parallelism_coef += 1
            for child in node.children:
                tmp_stack.append(child)
        parallelism_degree.append(parallelism_coef)
        stack = tmp_stack.copy()

    return parallelism_degree


def metric_vertex_count(wg: WorkGraph) -> float:
    return wg.vertex_count


def metric_average_work_per_activity(wg: WorkGraph) -> float:
    return sum(node.work_unit.volume for node in wg.nodes) / wg.vertex_count


def metric_max_children(wg: WorkGraph) -> float:
    return max((len(node.children) for node in wg.nodes if node.children))


def metric_average_resource_usage(wg: WorkGraph) -> float:
    return sum(sum((req.min_count + req.max_count) / 2 for req in node.work_unit.worker_reqs)
               for node in wg.nodes) / wg.vertex_count


def metric_max_parents(wg: WorkGraph) -> float:
    return max((len(node.parents) for node in wg.nodes if node.parents))


def encode_graph(wg: WorkGraph) -> list[float]:
    return [
        metric_vertex_count(wg),
        metric_average_work_per_activity(wg),
        metric_max_children(wg),
        metric_average_resource_usage(wg),
        metric_max_children(wg),
        *metric_graph_parallelism_degree(wg)
    ]
