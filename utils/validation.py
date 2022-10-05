import json
import pickle
from typing import Dict, Set, List

from generator.enviroment.contractor import get_contractor_with_equal_proportions
from schemas.schedule import Schedule
from schemas.contractor import Contractor, AgentsDict
from schemas.works import WorkUnit
from schemas.graph import GraphNode, WorkGraph


def build_adjacency_graph(graph: WorkGraph) -> (Dict[str, Set[str]], Dict[str, WorkUnit]):
    """
    Create adjacency list from work graph
    Create requirements dict with work and all descriptions of work
    :param graph:
    :return:
    """
    adjacency_graph = {}
    visited = set()
    requirements_dict = {}
    dict_for_building = []
    required_masters = []

    def dfs(graph_wg: WorkGraph, start: GraphNode, visited_node: Set[str]) -> None:
        if start.work_unit.id not in visited_node:
            adjacency_graph[start.work_unit.id] = set()
            requirements_dict[start.work_unit.id] = start.work_unit
            required_masters.append(start.work_unit.worker_reqs)
            visited_node.add(start.work_unit.id)
            for i in range(len(start.children)):
                adjacency_graph[start.work_unit.id].add(start.children[i].work_unit.id)
                dict_for_building.append((start, start.children[i].work_unit.id))
                dfs(graph_wg, start.children[i], visited_node)

    dfs(graph, graph.start, visited)

    return adjacency_graph, requirements_dict


def reverse_dict(adj_list: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Create dict work: set of parent work
    :param adj_list:
    :return: reverse adj list
    """
    result_rev_adj_list = {}
    for key in adj_list:
        for val in adj_list[key]:
            result_rev_adj_list[val] = result_rev_adj_list.get(val, tuple()) + (key,)
    return result_rev_adj_list


def check_validity(rev_adj_list_graph: Dict[str, Set[str]], schedule: Schedule) -> bool:
    """
    Check valid od order. Check if parent work already in dict with completed works
    :param rev_adj_list_graph:
    :param schedule:
    :return: bool
    """
    already_completed = set()
    for w in schedule.works:
        w_id = w.work_unit.id
        pred_jobs = rev_adj_list_graph.get(w_id, set())
        if any(pred_jobs.difference(already_completed)):
            return False
        already_completed.add(w_id)
    return True


def check_resources(schedule: Schedule, wg: WorkGraph, agents: AgentsDict) -> bool:
    """
    Check numbers of workers in any time moment doesn't increase total number of available workers
    :param schedule:
    :param agents:
    :param wg:
    :return: bool
    """
    dict_moment: Dict[int, Dict[str, Dict[str, int]]] = {}
    # go throughout scheduling
    for scheduled_work in schedule.works:
        start, end = tuple(sorted(scheduled_work.start_end_time))
        # we should check and count only inseparable parents
        if wg[scheduled_work.work_unit.id].is_inseparable_son():
            continue
        # take all int time moments in current work
        for t in range(int(start), int(end)):
            # add to dict moment of time and agents
            dict_moment_t: Dict[str, Dict[str, int]] = dict_moment.get(t, None)
            if dict_moment_t is None:
                dict_moment[t] = dict_moment_t = {name_worker:
                                                  {cont: 0 for cont in contractors2agents.keys()}
                                                  for name_worker, contractors2agents in agents.items()}
            # add to this time moment numbers of workers for each type, who participate in this work
            for worker in scheduled_work.workers:
                dict_moment_t[worker.name][worker.contractor_id] = cur = \
                    dict_moment_t[worker.name][worker.contractor_id] + worker.count
                if agents[worker.name][worker.contractor_id].count < cur:
                    print(agents[worker.name], cur, worker.name, t)
                    return False
    return True


def check_validity_of_scheduling(schedule: Schedule,
                                 agents: AgentsDict,
                                 work_graph: WorkGraph) -> None:
    """
    Check scheduling
    1. Valid order
    2. In every time moment to check numbers of appointed resource are not increase
    the total numbers of workers of each type
    Raise an assertion error
    :param schedule:
    :param agents:
    :param work_graph:
    :return: None
    """
    adj_list_graph, requirements_dict = build_adjacency_graph(work_graph)
    rev_adj_list_graph = reverse_dict(adj_list_graph)
    check_order = check_validity(rev_adj_list_graph, schedule)
    check_res = check_resources(schedule, work_graph, agents)
    assert check_res
    assert check_order


if __name__ == '__main__':
    filepath_wg = "../../resources/wide_graph.pickle"
    time_units = 'day'

    with open(filepath_wg, "rb") as f:
        data = pickle.load(f)
    wg: WorkGraph = data["work_graph"]

    agents_from_user = {'driver': 58, 'fitter': 65, 'handyman': 80, 'electrician': 19, 'manager': 29, 'engineer': 19}

    contractors: List[Contractor] = get_contractor_with_equal_proportions(5)

    filepath_scheduler = "../scheduler_heft.json"
    with open(filepath_scheduler) as f:
        json_scheduler = json.load(f)

    # TODO Rewrite agents_from_user value
    check_validity_of_scheduling(json_scheduler, agents_from_user, wg)
