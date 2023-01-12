from copy import deepcopy
from itertools import chain
from random import Random
from typing import Callable

from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.requirements import WorkerReq


def _extend_str_fields(new_uniq_count: int, wg: WorkGraph, rand: Random,
                       uniq_finder_str: Callable[[WorkGraph], list[str]],
                       update_node: Callable[[GraphNode, dict[str, list[str]], Random], None]) -> WorkGraph:
    uniq_str = uniq_finder_str(wg)
    if new_uniq_count <= len(uniq_str):
        return wg
    wg = deepcopy(wg)
    rand.shuffle(uniq_str)
    names_plus_one = set(uniq_str[:new_uniq_count % len(uniq_str)])
    suffixes = [' #' + str(i + 1) for i in range(new_uniq_count // len(uniq_str))]
    name_to_suffixes = {name: (suffixes + (['_0'] if name in names_plus_one else [])) for name in uniq_str}
    [update_node(node, name_to_suffixes, rand) for node in wg.nodes]
    return wg


def _get_uniq_resource_kinds(wg: WorkGraph) -> list[str]:
    uniq_res_kinds = list(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
    return uniq_res_kinds


def _update_resource_names(node: GraphNode, name_to_suffixes: dict[str, list[str]], rand: Random):
    new_req: list[WorkerReq] = []
    for res_req in node.work_unit.worker_reqs:
        kind = res_req.kind
        options = name_to_suffixes[kind]
        kind = kind + options[rand.randint(0, len(options)-1)]
        new_req.append(WorkerReq(kind, res_req.volume, res_req.min_count, res_req.max_count, res_req.name))
    node.work_unit.worker_reqs = new_req


def _get_uniq_work_names(wg: WorkGraph) -> list[str]:
    start_finish_names = {wg.start.work_unit.name, wg.finish.work_unit.name}
    uniq_work_names = list(set(node.work_unit.name for node in wg.nodes) - start_finish_names)
    return uniq_work_names


def _update_work_name(node: GraphNode, name_to_suffixes: dict[str, list[str]], rand: Random) -> None:
    name = node.work_unit.name
    if name in name_to_suffixes:
        options = name_to_suffixes[name]
        node.work_unit.name = name + options[rand.randint(0, len(options)-1)]


def extend_resources(uniq_resources: int, wg: WorkGraph, rand: Random) -> WorkGraph:
    """
    Increases the number of unique resources in WorkGraph
    :param uniq_resources: the amount to which you need to increase
    :param wg: original WorkGraph
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :return: modified WorkGraph
    """
    return _extend_str_fields(uniq_resources, wg, rand, _get_uniq_resource_kinds, _update_resource_names)


def extend_names(uniq_activities: int, wg: WorkGraph, rand: Random) -> WorkGraph:
    """
     Increases the number of unique work names in WorkGraph
    :param uniq_activities:  the amount to which you need to increase
    :param wg: original WorkGraph
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :return: modified WorkGraph
    """
    return _extend_str_fields(uniq_activities, wg, rand, _get_uniq_work_names, _update_work_name)
