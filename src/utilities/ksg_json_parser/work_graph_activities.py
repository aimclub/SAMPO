import json
import os
from collections import Counter
from itertools import chain
from typing import List, Dict, Any, Tuple, Callable, Optional
from typing import Set

from matplotlib import pyplot as plt

from generator.pipeline.cluster import get_finish_stage
from generator.pipeline.cluster import get_start_stage
from schemas.works import WorkUnit
from schemas.graph import GraphNode, WorkGraph
from utilities.ksg_json_parser.json_to_mongo_format import ActivityType
from utilities.visualization.work_graph import work_graph_fig

Activity = Dict['str', Any]


def preprocessing(activities: List[Activity]):
    id_to_activity = {}
    edges_count = {a['activity_id']: 0 for a in activities}
    ids_set = set(list(edges_count.keys()))
    activity_successors = set()
    activity_pred = set()

    for activity in activities:
        a_id = activity['activity_id']

        activity['pred_ids'] = set([pred['pred_activity_id'] for pred in activity['activity_predessors']]) & ids_set
        activity['succ_ids'] = set([pred['succ_activity_id'] for pred in activity['activity_successors']]) & ids_set

        id_to_activity[a_id] = activity
        edges_count[a_id] = len(activity['pred_ids'])

        activity_pred |= activity['pred_ids']
        activity_successors |= activity['succ_ids']
    return id_to_activity, edges_count, activity_successors, activity_pred


def make_node_unit(json_activity: Activity, id_to_node: Dict[str, GraphNode], start_node: GraphNode,
                   wbs_names: Dict[Tuple[str, str], str],
                   word_count: Optional[Tuple[int, int]] = (1, 1)) -> (
        GraphNode, Set[str], Set[str]):
    js = json_activity
    project_id = js['project_id'] if 'project_id' in js else 'default'
    activity_wbs_id = js['activity_wbs_id'] if 'activity_wbs_id' in js else 'default'
    pair_id = (project_id, activity_wbs_id)
    group = activity_wbs_id if pair_id not in wbs_names else wbs_names[pair_id]
    # group = ' '.join(js['activity_name'].split()[:word_count[1]])
    # gropu = ' '.join(js['activity_name'].split()[:word_count[1]])
    w = WorkUnit(id=js['activity_id'], name=f"{' '.join(js['activity_name'].split()[:word_count[0]])}", worker_reqs=[],
                 group=group)

    parents = [id_to_node[pred_id] for pred_id in js['pred_ids'] if pred_id in id_to_node]
    node = GraphNode(w, parents or [start_node])
    id_to_node[js['activity_id']] = node
    return node, js['pred_ids'], js['succ_ids']


def build_graph(activities: List[Activity], id_to_activity: Dict[str, Activity], edges_count: Dict[str, int],
                wbs_names: Dict[Tuple[str, str], str],
                word_count: Optional[Tuple[int, int]] = (1, 1)):
    start_node = get_start_stage()

    id_to_a = id_to_activity
    id_to_node: Dict[str, GraphNode] = {}

    used: Set[str] = set()
    has_succ: Set[str] = set()

    while True:
        cur_activities_ids: List[str] = [a['activity_id'] for a in activities
                                         if edges_count[a['activity_id']] == 0 and a['activity_id'] not in used]
        used |= set(cur_activities_ids)
        # print([(key, edges_count[key]) for key in cur_activities_ids], "\n\n")
        if len(cur_activities_ids) == 0:
            break
        work_units, predessors_ids, successors_ids = list(
            zip(*[make_node_unit(id_to_a[key], id_to_node, start_node, wbs_names, word_count)
                  for key in cur_activities_ids]))
        counter = Counter(list(chain(*successors_ids)))
        predessors_ids_set = set(list(chain(*predessors_ids)))
        edges_count = {key: (val if key not in counter else val - counter[key]) for key, val in edges_count.items()}
        has_succ |= predessors_ids_set

    final_stage = [id_to_node[key] for key in id_to_node if key not in has_succ]
    end_node = get_finish_stage(final_stage)
    graph = WorkGraph(start_node, end_node)
    return graph, has_succ, used


def plot_activities(activities: List[ActivityType], plot_func: Callable, wbs_names: Dict[Tuple[str, str], str],
                    path: Optional[str] = "work_graph.png",
                    fig_size: Tuple[int, int] = (28, 17), fig_dpi: int = 300,
                    word_count: Optional[Tuple[int, int]] = (1, 1),
                    show_arrows: Optional[bool] = True, show_names: Optional[bool] = False):
    g_id_to_activity, g_edges_count, _, _ = preprocessing(activities)
    work_graph, has_succ_activities, used_activities = build_graph(activities, g_id_to_activity, g_edges_count,
                                                                   wbs_names, word_count)
    service_nodes = [work_graph.start.id, work_graph.finish.id]
    # work_graph = [node for node in work_graph.start.children if node.id == '210823'][0]
    work_graph_fig(work_graph, fig_size, show_arrows=show_arrows, hide_node_ids=service_nodes, fig_dpi=fig_dpi,
                   show_names=show_names)
    plot_func(fname=path, format="png")


if __name__ == "__main__":
    data_path = "../../../../resources/ksg_data/"
    with open(data_path + 'new_port_projects_union.json') as json_file:
        ksg_info = json.load(json_file)
    p_wbs_ids = chain(*[[(p['project_id'], wbs['wbs_id'], wbs['wbs_name']) for wbs in p['wbs_tasks']] for p in
                        ksg_info[0]['projects']])
    wbs_names: Dict[Tuple[str, str], str] = {(p_id, wbs_id): wbs_name for p_id, wbs_id, wbs_name in p_wbs_ids}
    image_path = "../../../../images/ksg_msh/"
    activities_names = sorted([file for file in os.listdir(data_path)
                               if "_activities.json" in file and "_union" not in file])
    # activities_names = ['обустройство_инфрастр&АД_ЦПС_1_activities.json', 'обустройство_инфрастр&АД_ЦПС_2_activities.json']#, 'ОКП-2&4-ГАЗ_activities.json']
    # activities_names = ['ОКП-1&4-ОК7.1_activities.json']
    processed_act = set()  # set([file.replace('.png', '.json') for file in os.listdir(image_path)])
    json_activities = []
    for name in activities_names:
        if name in processed_act:
            continue
        with open(data_path + name) as json_file:
            data = json.load(json_file)
        json_activities = data['activities']
        if len(json_activities) == 0 or json_activities[0]['field_id'] != 734195:
            continue
        # for ind in range(len(json_activities)):

        print(len(json_activities), ' ', name)
        plot_activities(json_activities, plt.savefig, wbs_names, image_path + name.replace('.json', '.png'), fig_size=(24, 16),
                        fig_dpi=300, word_count=(5, 3), show_names=False)
        # plot_activities(json_activities, plt.savefig, image_path + name.replace('.json', '.png'), fig_size=(24, 16), fig_dpi=300)
