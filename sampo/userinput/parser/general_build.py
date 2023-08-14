import math
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


def break_loops_in_input_graph(works_info: pd.DataFrame) -> pd.DataFrame:
    """
    Syntax sugar

    :param works_info:
    :return:
    """

    tasks_ahead = ['-1']
    print(works_info)
    for _, row in works_info.iterrows():
        predecessor_ids_copy = row['predecessor_ids'].copy()
        connection_types_copy = row['connection_types'].copy()
        lags_copy = row['lags'].copy()
        for pred_id in row['predecessor_ids']:
            if pred_id in tasks_ahead:
                continue
            index = predecessor_ids_copy.index(pred_id)
            predecessor_ids_copy.pop(index)
            connection_types_copy.pop(index)
            lags_copy.pop(index)
        works_info.at[_, 'predecessor_ids'] = ','.join(map(str, predecessor_ids_copy))
        works_info.at[_, 'connection_types'] = ','.join(map(str, connection_types_copy))
        works_info.at[_, 'lags'] = ','.join(map(str, lags_copy))
        tasks_ahead.append(str(_))

    return works_info


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
    circuits = find_all_circuits(works_info)

    for _, row in works_info[::-1].iterrows():
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
                              none_elem: Any | None = NONE_ELEM) -> pd.Series:
    new_column = column.copy().astype(str).apply(
        lambda elems: [cast(elem) if elem != '' and elem != str(math.nan) else none_elem for elem in
                       elems.split(',')] if (elems != str(math.nan) and elems.split(',') != '') else [none_elem])
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

    # frame = break_loops_in_input_graph(frame)

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
    """
    Sort DataFrame of works in topological order

    :param frame: DataFrame of works
    :return: topologically sorted DataFrame
    """
    G = nx.DiGraph()
    for _, row in frame.iterrows():
        G.add_node(row['activity_id'])
        for pred_id, con_type, lag in row['edges']:
            G.add_edge(pred_id, row['activity_id'])

    sorted_nodes = list(nx.topological_sort(G))
    frame['sort_key'] = frame['activity_id'].apply(lambda x: sorted_nodes.index(x))
    frame = frame.sort_values('sort_key')

    return frame


def build_work_graph(frame: pd.DataFrame, resource_names: list[str]) -> WorkGraph:
    start = get_start_stage()
    has_succ = set()
    id_to_node = {NONE_ELEM: start}

    for _, row in frame.iterrows():
        if 'min_req' in frame.columns and 'max_req' in frame.columns:
            reqs = [WorkerReq(res_name, row[res_name],
                              row['min_req'][res_name],
                              row['max_req'][res_name]
                              ) for res_name in resource_names
                    if 0 < row['min_req'][res_name] <= row['max_req'][res_name]]
        else:
            reqs = [WorkerReq(kind=res_name,
                              volume=row[res_name],
                              min_count=int(row[res_name] / 3),
                              max_count=math.ceil(row[res_name] * 10))
                    for res_name in resource_names
                    if row[res_name] > 0]
        is_service_unit = len(reqs) == 0
        work_unit = WorkUnit(row['activity_id'], row['activity_name'], reqs, group=row['activity_name'],
                             volume=row['volume'], volume_type=row['measurement'], is_service_unit=is_service_unit,
                             display_name=row['granular_name'])
        has_succ |= set(row['edges'][0])
        parents = [(id_to_node[p_id], lag, conn_type) for p_id, conn_type, lag in row.edges]
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
