import hashlib
import queue
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Optional, Callable

import pandas as pd
from matplotlib import pyplot as plt, patches, axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from sampo.schemas.graph import GraphNode, WorkGraph
from sampo.schemas.works import WorkUnit

X_STEP = 0.8
X_DELTA = 0.2
X_PERIOD = X_STEP + X_DELTA

TEXT_X_DELTA = 0.01
TEXT_Y_DELTA = 0.05

Y_STEP = 0.7
Y_DELTA = 0.3
Y_PERIOD = Y_STEP + Y_DELTA

MIN_LEN = 0.2
ARROW_HEAD_WIDTH = Y_STEP / 2
ARROW_HEAD_LENGTH = X_DELTA / 10
BORDER_LINE_WIDTH = 0.3
LINE_WIDTH = 0.6

INF_INT = int(1e9)

SIZE_LIMIT = int(1e3)
DEFAULT_DPI = 50
DPI_LIMIT = 3e5 / SIZE_LIMIT


def work_graph_fig(graph: WorkGraph or GraphNode, fig_size: Tuple[int, int],
                   fig_dpi: Optional[int] = 300,
                   max_deep: Optional[int] = None,
                   show_names: Optional[bool] = False, show_arrows: Optional[bool] = True,
                   hide_node_ids: Optional[List[str]] = None,
                   legend_shift: Optional[int] = 0,
                   show_only_not_dotted: Optional[bool] = False,
                   dotted_edges: Set[Tuple[str, str]] = None,
                   black_list_edges: Set[Tuple[str, str]] = None,
                   jobs2text_function: Optional[Callable[[pd.Series], str]] = None,
                   text_size: Optional[int] = 1) -> Figure:
    start = graph.start if type(graph) == WorkGraph else graph
    jobs, id_to_job, colors = collect_jobs(start, max_deep)
    jobs = setup_jobs(start, jobs, id_to_job)
    dotted_edges = dotted_edges or set()
    black_list_edges = black_list_edges or set()
    hide_node_ids = set(hide_node_ids or [])

    df = pd.DataFrame(jobs)
    df = df.sort_values(by=["cluster", "group"])

    fig, ax = plt.subplots(1, figsize=fig_size, dpi=fig_dpi)
    ax_add_works(ax, df, show_names, hide_node_ids, jobs2text_function, text_size)
    if show_arrows:
        ax_add_dependencies(ax, df, id_to_job, dotted_edges, hide_node_ids, show_only_not_dotted, black_list_edges)

    ax.set_xlim(df.start.min() - X_DELTA, df.finish.max() + X_DELTA + legend_shift)
    ax.set_ylim(df.y_position.min() - Y_PERIOD, df.y_position.max() + 2 * Y_PERIOD)

    groups = df.group.unique()
    legend_elements = [Patch(facecolor=color_from_str(g_name), label=g_name) for g_name in groups]
    plt.legend(handles=legend_elements, loc='upper right')
    return fig


def extract_cluster_name(work_name: str) -> str:
    cluster_name = work_name.split("`")
    cluster_name = "" if len(cluster_name) < 2 else cluster_name[1]
    return cluster_name


def calculate_work_volume(work_unit: WorkUnit) -> float:
    volume = sum([req.volume for req in work_unit.worker_reqs])
    return volume


def collect_jobs(start: GraphNode, max_deep: Optional[int] = None) -> (List[Dict], Dict[str, int], Dict[str, str]):
    max_deep = max_deep or INF_INT
    q = queue.Queue()
    q.put((0, start))
    id_to_job: Dict[str, int] = dict()
    used: Set[str] = {start.id}
    jobs: List[Dict] = []
    colors: Dict[str, str] = dict()
    max_volume: float = 0

    while not q.empty():
        deep, node = q.get()
        if deep >= max_deep:
            continue
        unit = node.work_unit
        volume = calculate_work_volume(unit)
        max_volume = max(max_volume, volume)
        id_to_job[unit.id] = len(jobs)
        jobs.append(
            dict(job_id=len(jobs), work_id=str(unit.id), task=unit.name, start=X_PERIOD * deep,
                 children=node.children, parents=node.parents,
                 group=unit.group, color=color_from_str(unit.group), volume=volume,
                 cluster=extract_cluster_name(unit.name)))
        colors[unit.group] = color_from_str(unit.group)
        for c in node.children:
            work_id = c.id
            if work_id in used:
                continue
            q.put((deep + 1, c))
            used.add(work_id)
    return jobs, id_to_job, colors


def setup_jobs(start: GraphNode, jobs: List[Dict], id_to_job: Dict[str, int]) -> List[Dict]:
    cluster_deep_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    not_used_parents = {job["job_id"]: len(job["parents"]) for job in jobs}

    q = queue.Queue()
    start_ind = id_to_job[start.id]
    q.put((0, start_ind))

    while not q.empty():
        deep, ind = q.get()
        jobs[ind]["start"] = deep * X_PERIOD
        jobs[ind]["finish"] = jobs[ind]["start"] + X_STEP
        cluster_deep_counts[jobs[ind]["cluster"]][jobs[ind]["start"]] += 1

        for node in jobs[ind]["children"]:
            if node.id not in id_to_job:
                continue
            child_ind = id_to_job[node.id]
            not_used_parents[child_ind] -= 1
            if not_used_parents[child_ind] <= 0:
                q.put((deep + 1, child_ind))

    max_y_pos: Dict[int, int] = dict()

    def dfs(v_ind: int, y_pos: int) -> int:
        jobs[v_ind]["y_position"] = y_pos

        used = [id_to_job[child.id] for child in jobs[v_ind]['children'] if id_to_job[child.id] in max_y_pos]
        y_pos = max([y_pos] + [max_y_pos[c_ind] for c_ind in used] or [y_pos])

        for child in jobs[v_ind]['children']:
            c_ind = id_to_job[child.id]
            if c_ind not in max_y_pos:
                y_pos = dfs(c_ind, y_pos) + 1

        max_y_pos[v_ind] = max(jobs[v_ind]["y_position"], y_pos - 1)
        return max_y_pos[v_ind]

    _ = dfs(start_ind, 0)
    return jobs


def ax_add_works(ax: axes.Axes, df: pd.DataFrame, show_names: bool, hide_nodes_id: Set[str],
                 jobs2text_function: Optional[Callable[[pd.Series], str]] = None, text_size: Optional[int] = 1):
    if jobs2text_function is None:
        jobs2text_function = default_job2text
    if not show_names:
        jobs2text_function = empty_job2text

    for ind, row in df.iterrows():
        if row.work_id in hide_nodes_id:
            continue
        x: float = float(row.start)
        y: float = float(row.y_position)
        length: float = row.finish - row.start
        rect = patches.Rectangle((x, y), length, Y_STEP, linewidth=BORDER_LINE_WIDTH, edgecolor="k",
                                 facecolor=row.color)
        ax.add_patch(rect)
        ax.text(x + TEXT_X_DELTA, y + TEXT_Y_DELTA, jobs2text_function(row), fontsize=text_size)


def ax_add_dependencies(ax: axes.Axes, df: pd.DataFrame, id_to_job: Dict[str, int],
                        dotted_edges: Set[Tuple[str, str]], hide_nodes_id: Set[str], show_only_not_dotted: bool,
                        black_list_edges: Set[Tuple[str, str]]):
    for ind, job in df.iterrows():
        if job.work_id in hide_nodes_id:
            continue
        for son in job.children:
            if son.id in hide_nodes_id or son.id not in id_to_job:
                continue
            linestyle = 'dotted' if ((str(job.work_id), str(son.id)) in dotted_edges) else None
            if linestyle == 'dotted' and show_only_not_dotted or (str(job.work_id), str(son.id)) in black_list_edges:
                continue
            draw_arrow_between_jobs(ax, job, df[df.job_id == id_to_job[son.id]].iloc[0], linestyle=linestyle)


def draw_arrow_between_jobs(ax, first_job_dict, second_job_dict, linestyle: Optional[str] = None):
    x_fj = first_job_dict.finish
    y_fj = first_job_dict.y_position + Y_STEP / 2
    x_mid = second_job_dict.start - X_DELTA / 2

    x_sj = second_job_dict.start
    y_sj = second_job_dict.y_position + Y_STEP / 2
    line_color = middle_color(first_job_dict["color"], second_job_dict['color'])

    ax.plot([x_fj, x_mid, x_mid], [y_fj, y_fj, y_sj], color=line_color, linewidth=LINE_WIDTH, linestyle=linestyle)
    ax.arrow(x_mid, y_sj, x_sj - x_mid - ARROW_HEAD_LENGTH, 0, color=line_color, linewidth=LINE_WIDTH, shape="full",
             head_width=ARROW_HEAD_WIDTH,
             head_length=ARROW_HEAD_LENGTH, linestyle=linestyle)
    return


def color_from_str(name: str) -> str:
    name = name or ''
    hashed_string = hashlib.sha256(name.encode())
    rgb_code = hashed_string.hexdigest()[len(hashed_string.hexdigest()) - 6:]
    return f"#{rgb_code}"


def middle_color(color_a: str, color_b: str) -> str:
    mid = "#"
    for i in range(1, 7, 2):
        a = int(color_a[i:i + 2], 16)
        b = int(color_b[i:i + 2], 16)
        c = int((a + b) / 2 * 0.7)
        mid += "%0.2X" % c
    return mid


def default_job2text(row: pd.Series) -> str:
    return row.work_id + " " + row.task


def empty_job2text(row: pd.Series) -> str:
    return ''
