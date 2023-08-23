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
                              none_elem: Any | None = NONE_ELEM) -> pd.Series:
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
    existed_ids = set(frame['activity_id'])

    def filter_predecessor_info(row):
        filtered_predecessors = [(pred_id, con_type, lag)
                                 for pred_id, con_type, lag in zip(row['predecessor_ids'], row['connection_types'], row['lags'])
                                 if pred_id in existed_ids]
        if not filtered_predecessors:
            filtered_predecessors.append((NONE_ELEM, EdgeType.FinishStart, float(NONE_ELEM)))
        return filtered_predecessors

    frame['edges'] = frame.apply(filter_predecessor_info, axis=1)
    frame.drop(['predecessor_ids', 'connection_types', 'lags'], axis=1, inplace=True)

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
                             display_name=row['activity_name'])
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
