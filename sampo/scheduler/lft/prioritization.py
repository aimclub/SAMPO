from sampo.scheduler.lft.time_computaion import work_min_max_duration
from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.time_estimator import WorkTimeEstimator
from operator import itemgetter
import random


def lft_prioritization(wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
    """
    Return ordered nodes by MIN-LFT.
    """

    # def is_eligible(node_id):
    #     return all([pred_id in selected_ids for pred_id in parents[node_id]])

    # inverse weights
    nodes2min_max_duration = {node.id: work_min_max_duration(node, work_estimator)
                              for node in wg.nodes}
    min_project_max_duration = sum([durations[0] for durations in nodes2min_max_duration.values()])
    max_project_max_duration = sum([durations[1] for durations in nodes2min_max_duration.values()])
    nodes2lft = {wg.finish.id: (min_project_max_duration, max_project_max_duration)}
    nodes2lst = {wg.finish.id: (nodes2lft[wg.finish.id][0] - nodes2min_max_duration[wg.finish.id][0],
                                nodes2lft[wg.finish.id][1] - nodes2min_max_duration[wg.finish.id][1])}
    for node in reversed(wg.nodes[:-1]):
        suc_lst = [nodes2lst[suc.id] for suc in node.children]
        nodes2lft[node.id] = (min(suc_lst, key=itemgetter(0))[0], min(suc_lst, key=itemgetter(1))[1])
        nodes2lst[node.id] = (nodes2lft[node.id][0] - nodes2min_max_duration[node.id][0],
                              nodes2lft[node.id][1] - nodes2min_max_duration[node.id][1])

    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    nodes2lft = {node.id: sum(nodes2lft[node.get_inseparable_chain_with_self()[-1].id]) / 2
                 for node in nodes}
    # nodes2lst = {node.id: sum(nodes2lst[node.get_inseparable_chain_with_self()[0].id]) / 2
    #              for node in nodes}

    ordered_nodes = sorted(nodes2lft.keys(), key=lambda id: nodes2lft[id], reverse=True)
    ordered_nodes = [wg.dict_nodes[id] for id in ordered_nodes]

    # inseparable_parents = {}
    # for node in nodes:
    #     for child in node.get_inseparable_chain_with_self():
    #         inseparable_parents[child.id] = node.id
    #
    # # here we aggregate information about relationships from the whole inseparable chain
    # children = {node.id: set([inseparable_parents[child.id]
    #                           for inseparable in node.get_inseparable_chain_with_self()
    #                           for child in inseparable.children]) - {node.id}
    #             for node in nodes}
    #
    # parents = {node.id: set() for node in nodes}
    # for node, node_children in children.items():
    #     for child in node_children:
    #         parents[child].add(node)
    #
    # selected_ids = []
    # candidates = {wg.start.id}
    #
    # while candidates:
    #     eligibles = [node_id for node_id in candidates if is_eligible(node_id)]
    #
    #     priority_mapper = nodes2lft if random.random() < 0.5 else nodes2lst
    #
    #     priorities = [priority_mapper[node_id] for node_id in eligibles]
    #     max_priority = max(priorities)
    #     deviations = [max_priority - priority + 1 for priority in priorities]
    #     total = sum(deviations)
    #     weights = [dev / total for dev in deviations]
    #
    #     selected_id = random.choices(eligibles, weights=weights)[0]
    #
    #     selected_ids.append(selected_id)
    #     candidates.remove(selected_id)
    #     candidates.update([suc_id for suc_id in children[selected_id]])
    #
    # ordered_nodes = list(reversed([wg.dict_nodes[node_id] for node_id in selected_ids]))

    # ordered_nodes = list(reversed(nodes))

    return ordered_nodes
