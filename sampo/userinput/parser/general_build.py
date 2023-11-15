import math
from collections import defaultdict
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


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, weight = None):
        self.graph[u].append((v, weight))

    def dfs_cycle(self, u, visited):
        stack = [u]

        while len(stack) > 0:
            v = stack.pop()
            if visited[v] == 1:
                visited[v] = 2
                continue
            if visited[v] == 2:
                continue
            visited[v] = 1
            stack.append(v)
            for neighbour, weight in self.graph[v]:
                if visited[neighbour] == 0:
                    stack.append(neighbour)
                elif visited[neighbour] == 1:
                    return True
        return False

    def find_cycle(self):
        visited = {}
        for key, value in self.graph.items():
            for v in value:
                visited[v[0]] = 0
            visited[key] = 0
        path = []

        for key, value in visited.items():
            if value == 0:
                if self.dfs_cycle(key, visited):
                    v = key
                    path.append(v)
                    while v not in path[:-1]:
                        for n, _ in self.graph[v]:
                            if visited[n] == 1:
                                v = n
                                break
                        path.append(v)
                    for v in path:
                        visited[v] = 2
                    return path

        return None

    def eliminate_cycle(self, cycle):
        min_weight = float('inf')
        min_edge = None

        for i in range(len(cycle) - 1):
            u = cycle[i]
            v = cycle[i + 1]

            for neighbor, weight in self.graph[u]:
                if weight is None:
                    weight = 1
                if neighbor == v and weight <= min_weight:
                    min_weight = weight
                    min_edge = (u, v)

        u, v = min_edge
        self.graph[u] = [(neighbor, weight) for neighbor, weight in self.graph[u] if neighbor != v]

    def eliminate_cycles(self, is_eliminate_cycle: bool = True) -> list | None:
        cycle = self.find_cycle()
        cycles = []

        while cycle is not None:
            for i in range(len(cycle)):
                if cycle[i] == cycle[-1]:
                    cycle = cycle[i:]
                    break
            if not is_eliminate_cycle:
                cycles.append(cycle)
                self.eliminate_cycle(cycle)
            else:
                self.eliminate_cycle(cycle)
            cycle = self.find_cycle()
        if not is_eliminate_cycle:
            if cycles:
                return cycles
            return None
        for v in self.graph:
            self.graph[v] = [u for u, w in self.graph[v]]


def break_loops_in_input_graph(works_info: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminate all cycles in received work graph. Algo breaks cycle by removing edge with the lowest weight
    (e.x. frequency of occurrence of the link in historical data).

    :param works_info: given work info
    :return: work info without cycles
    """
    graph = Graph()
    for _, row in works_info.iterrows():
        for pred_id, con_type, lag, counts in zip(row['predecessor_ids'], row['connection_types'], row['lags'],
                                                  row['counts']):
            if pred_id == '-1':
                continue
            graph.add_edge(pred_id, row['activity_id'], counts)

    graph.eliminate_cycles()
    for _, row in works_info.iterrows():
        i = 0
        while i < len(row['predecessor_ids']):
            if row['predecessor_ids'][i] not in graph.graph:
                i += 1
                continue
            if row['activity_id'] in graph.graph[row['predecessor_ids'][i]]:
                i += 1
                continue
            del row['predecessor_ids'][i]
            del row['connection_types'][i]
            del row['lags'][i]
            del row['counts'][i]
    return works_info.drop(columns=['counts'])


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

    return frame


def add_graph_info(frame: pd.DataFrame) -> pd.DataFrame:
    existed_ids = set(frame['activity_id'])

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

    frame['edges'] = frame[['predecessor_ids', 'connection_types', 'lags']].apply(lambda row: list(zip(*row)), axis=1)
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
        work_unit = WorkUnit(row['activity_id'], row['granular_name'], reqs, group=row['activity_name'],
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
