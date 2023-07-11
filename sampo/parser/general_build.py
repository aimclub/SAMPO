import math
import queue
from typing import Callable, Any
from uuid import uuid4

import networkx as nx
import pandas as pd

from sampo.generator.pipeline.project import get_start_stage, get_finish_stage
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.works import WorkUnit

UNKNOWN_CONN_TYPE = 0
NONE_ELEM = '-1'


def break_circuits_in_input_work_info(works_info: pd.DataFrame) -> pd.DataFrame:
    """
    The function breaks circuits:
    - find all circuits,
    - break first edge in circle, for example,

    we have edges:
    (1-2), (2-3), (3-4), (4-2)
    function deletes edge (2-3)

    :param works_info: dataframe, that contains information about works
    :return: cleaned work_info
    """
    circuits: list[list[str]] = find_all_circuits(works_info)

    for _, row in enumerate(works_info.index[::-1]):
        for cycle in circuits:
            inter = list(set(row['predecessor_ids']) & set(cycle))
            if len(inter) != 0:
                index = row['predecessor_ids'].index(inter[0])
                row['predecessor_ids'].pop(index)
                row['connection_types'].pop(index)
                row['lags'].pop(index)

    return works_info


def find_all_circuits(works_info: pd.DataFrame) -> list[list[str]]:
    """
    The function find all elementary circuits using the algorithm of Donald B. Johnson
    doi: 10.1137/0205007

    :param works_info: dataframe, that contains information about works
    :return: list of cycles
    """
    graph = nx.DiGraph()
    edges = []

    for _, row in works_info[::-1].iterrows():
        v = row['activity_id']
        for u in row['predecessor_ids']:
            edges.append((v, u))

    graph.add_nodes_from(list(works_info['activity_id']))
    graph.add_edges_from(edges)

    return list(nx.simple_cycles(graph))


def fix_df_column_with_arrays(column: pd.Series, cast: Callable[[str], Any] | None = str,
                              none_elem: Any | None = NONE_ELEM):
    new_column = column.copy().astype(str).apply(
        lambda elems: [cast(elem) for elem in elems.split(',')] if elems != str(math.nan) else [none_elem])
    return new_column


def preprocess_graph_df(frame: pd.DataFrame) -> pd.DataFrame:
    def normalize_if_number(s):
        return str(int(float(s))) \
            if s.replace('.', '', 1).isdigit() \
            else s

    frame['activity_id'] = frame['activity_id'].astype(str)
    frame['volume'] = frame['volume'].astype(float)
    frame['predecessor_ids'] = fix_df_column_with_arrays(frame['predecessor_ids'], cast=normalize_if_number)
    frame['connection_types'] = fix_df_column_with_arrays(frame['connection_types'],
                                                          cast=EdgeType,
                                                          none_elem=EdgeType.FinishStart)
    if 'lags' not in frame.columns:
        frame['lags'] = [NONE_ELEM] * len(frame)
    frame['lags'] = fix_df_column_with_arrays(frame['lags'], float)

    frame = break_circuits_in_input_work_info(frame)

    return frame


def add_graph_info(frame: pd.DataFrame) -> pd.DataFrame:
    existed_ids = set(frame["activity_id"])

    predecessor_ids, connection_types, lags = [], [], []
    for _, row in frame[['predecessor_ids', 'connection_types', 'lags']].iterrows():
        predecessor_ids.append([])
        connection_types.append([])
        lags.append([])
        for index in range(len(row['predecessor_ids'])):
            if row['predecessor_ids'][index] in existed_ids:
                predecessor_ids[-1].append(row['predecessor_ids'][index])
                connection_types[-1].append(row['connection_types'][index])
                lags[-1].append(row['lags'][index])
        if len(predecessor_ids[-1]) == 0:
            predecessor_ids[-1].append(NONE_ELEM)
            connection_types[-1].append(EdgeType.FinishStart)
            lags[-1].append(float(NONE_ELEM))
    frame['predecessor_ids'], frame['connection_types'], frame['lags'] = predecessor_ids, connection_types, lags

    frame["edges"] = frame[['predecessor_ids', 'connection_types', 'lags']].apply(lambda row: list(zip(*row)), axis=1)
    return frame


def topsort_graph_df(frame: pd.DataFrame) -> pd.DataFrame:
    frame['predessors_sh'] = [tuple(zip(*list(zip(*edges))[:2])) for edges in frame["edges"]]
    frame['predessors_set'] = frame['predecessor_ids'].apply(lambda ps: set(ps))  # & existed_ids)
    id2row = {w_id: ind for w_id, ind in zip(frame['activity_id'], frame.index)}
    sort_keys: dict[str, int] = dict()
    used: set[str] = {NONE_ELEM}
    ind: int = 0
    q = queue.Queue()
    for w_id in frame['activity_id']:
        q.put(w_id)

    while not q.empty():
        w_id = q.get()
        row = frame.iloc[id2row[w_id]]
        if len(row['predessors_set'] - used) == 0:
            sort_keys[w_id] = ind
            used.add(w_id)
            ind += 1
        else:
            q.put(w_id)

    frame['sort_key'] = [sort_keys[w_id] for w_id in frame["activity_id"]]
    frame = frame.sort_values('sort_key')

    return frame


def build_work_graph(frame: pd.DataFrame, workers_max_count: dict[str, float]) -> WorkGraph:
    start = get_start_stage()
    has_succ = set()
    id_to_node = {NONE_ELEM: start}
    resources_columns = [col for col in frame.columns if col.endswith('_res_fact')]

    for _, row in frame.iterrows():
        reqs = [WorkerReq(col, row[col],
                          min(row[col + '_min'], workers_max_count[col]),
                          min(row[col + '_max'], workers_max_count[col]),
                          ) for col in resources_columns
                if 0 < row[col + '_min'] <= row[col + '_max']]
        work_unit = WorkUnit(row['activity_id'], row['activity_name'], reqs, group=row['activity_name'],
                             volume=row['volume'], volume_type=row['measurement'])
        has_succ |= set(row.predecessor_ids)
        parents = [id_to_node[p_id] for p_id in row.predecessor_ids]
        node = GraphNode(work_unit, parents)
        id_to_node[row['activity_id']] = node

    without_succ = list(set(id_to_node.keys()) - has_succ)
    without_succ = [id_to_node[index] for index in without_succ]
    end = get_finish_stage(without_succ)
    graph = WorkGraph(start, end)
    return graph


def get_graph_contractors(path: str, contractor_name: str | None = 'ООО "***"') -> (
        list[Contractor], dict[str, float]):
    contractor_id = str(uuid4())
    workers_df = pd.read_csv(path, index_col='name')
    workers = {(name, 0): Worker(str(uuid4()), name, count * 2, contractor_id)
               for name, count in zip(workers_df.index, workers_df['count'])}
    workers_max_count = workers_df['count'].to_dict()
    contractors = [Contractor(contractor_id, contractor_name, list(workers_max_count.keys()), [], workers, {})]
    return contractors, workers_max_count
