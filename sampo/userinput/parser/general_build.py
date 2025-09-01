"""Helpers for constructing work graphs from tabular input.

Помощники для построения графов работ из табличных данных.
"""

import math
from collections import defaultdict
from typing import Callable, Any
from uuid import uuid4
from ast import literal_eval

import networkx as nx
import numpy as np
import pandas as pd

from sampo.schemas import Time
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode, WorkGraph, EdgeType
from sampo.schemas.requirements import WorkerReq, ZoneReq
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.name_mapper import NameMapper

UNKNOWN_CONN_TYPE = 0
NONE_ELEM = '-1'


class Graph:
    """Simple directed graph for detecting and removing cycles.

    Простой ориентированный граф для обнаружения и удаления циклов.
    """

    def __init__(self) -> None:
        """Initialize an empty adjacency list.

        Инициализирует пустой список смежности.
        """

        self.graph = defaultdict(list)

    def add_edge(self, u, v, weight=None) -> None:
        """Add a directed edge to the graph.

        Добавляет ориентированное ребро в граф.

        Args:
            u: Source node.
                Исходный узел.
            v: Destination node.
                Конечный узел.
            weight (float | None): Optional edge weight.
                Необязательный вес ребра.
        """

        self.graph[u].append((v, weight))

    def dfs_cycle(self, u, visited):
        """Depth-first search to detect a cycle.

        Поиск в глубину для обнаружения цикла.

        Args:
            u: Start node.
                Начальный узел.
            visited: Visited state dictionary.
                Словарь состояний посещения.

        Returns:
            bool: ``True`` if a cycle is found.
                bool: ``True``, если обнаружен цикл.
        """

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
        """Find a cycle in the graph if it exists.

        Находит цикл в графе, если он существует.

        Returns:
            list | None: Sequence of nodes forming a cycle or ``None``.
                list | None: последовательность узлов цикла или ``None``.
        """

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

    def eliminate_cycle(self, cycle) -> None:
        """Remove the lightest edge of a given cycle.

        Удаляет наиболее лёгкое ребро заданного цикла.

        Args:
            cycle: Sequence of nodes representing a cycle.
                Последовательность узлов, образующих цикл.
        """

        min_weight = float("inf")
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
        """Iteratively remove all cycles in the graph.

        Итеративно удаляет все циклы в графе.

        Args:
            is_eliminate_cycle (bool, optional): Return removed cycles instead of
                eliminating them if ``False``. Defaults to ``True``.
                Если ``False``, вернуть удалённые циклы вместо их удаления.

        Returns:
            list | None: List of removed cycles or ``None``.
                list | None: список удалённых циклов или ``None``.
        """

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
    """Remove cycles from the input work graph.

    Удаляет циклы из входного графа работ.

    The algorithm deletes the edge with the smallest weight within each
    detected cycle (e.g., link frequency in history).

    Алгоритм удаляет ребро с наименьшим весом в каждом найденном цикле
    (например, частота связи в истории).

    Args:
        works_info (pd.DataFrame): Input work information.
            Входные данные о работах.

    Returns:
        pd.DataFrame: Work info without cycles.
            pd.DataFrame: данные о работах без циклов.
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
    """Convert comma-separated strings in a column to lists.

    Преобразует строки с разделителями в списки.

    Args:
        column (pd.Series): Column with comma-separated values.
            Колонка со значениями, разделёнными запятыми.
        cast (Callable[[str], Any] | None): Function for element conversion.
            Функция преобразования элемента.
        none_elem (Any | None): Placeholder for missing elements.
            Значение-заглушка для отсутствующих элементов.

    Returns:
        pd.Series: Converted column.
            pd.Series: преобразованная колонка.
    """

    new_column = column.copy().astype(str).apply(
        lambda elems: [cast(elem) if elem != '' and elem != str(math.nan) else none_elem for elem in
                       elems.split(',')] if (elems != str(math.nan) and elems.split(',') != '') else [none_elem])
    return new_column


def preprocess_graph_df(frame: pd.DataFrame,
                        name_mapper: NameMapper | None = None) -> pd.DataFrame:
    """Prepare work graph data for building.

    Подготавливает данные графа работ для построения.

    Args:
        frame (pd.DataFrame): Raw work information.
            Исходные данные о работах.
        name_mapper (NameMapper | None): Mapper of activity names.
            Сопоставитель названий работ.

    Returns:
        pd.DataFrame: Normalized DataFrame ready for processing.
            pd.DataFrame: нормализованный DataFrame для обработки.
    """

    def normalize_if_number(s):
        return str(int(float(s))) if s.replace('.', '', 1).isdigit() else s

    temp_lst = [math.nan] * frame.shape[0]

    for col in ['predecessor_ids', 'connection_types', 'lags', 'counts']:
        if col not in frame.columns:
            frame[col] = temp_lst

    if 'granular_name' not in frame.columns:
        frame['granular_name'] = [name_mapper[activity_name] for activity_name in frame['activity_name']]

    frame['activity_id'] = frame['activity_id'].astype(str)
    frame['volume'] = [float(x.replace(',', '.')) if isinstance(x, str) else float(x) for x in frame['volume']]

    if 'min_req' in frame.columns and 'max_req' in frame.columns:
        frame['min_req'] = [literal_eval(x) if isinstance(x, str) else x for x in frame['min_req']]
        frame['max_req'] = [literal_eval(x) if isinstance(x, str) else x for x in frame['max_req']]

    frame['predecessor_ids'] = fix_df_column_with_arrays(frame['predecessor_ids'], cast=normalize_if_number)
    frame['connection_types'] = fix_df_column_with_arrays(
        frame['connection_types'], cast=EdgeType, none_elem=EdgeType.FinishStart
    )
    if 'lags' not in frame.columns:
        frame['lags'] = [NONE_ELEM] * len(frame)
    frame['lags'] = fix_df_column_with_arrays(frame['lags'], float)
    for col in ['predecessor_ids', 'connection_types', 'lags', 'counts']:
        frame[col] = frame[col].astype(object)

    for _, row in frame.iterrows():
        frame.at[_, 'counts'] = [np.iinfo(np.int64).max] * len(frame.at[_, 'lags'])

    return frame


def add_graph_info(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter nonexistent predecessors and collect edge info.

    Отфильтровывает несуществующих предшественников и собирает информацию о рёбрах.

    Args:
        frame (pd.DataFrame): Preprocessed DataFrame.
            Предварительно обработанный DataFrame.

    Returns:
        pd.DataFrame: DataFrame enriched with edge tuples.
            pd.DataFrame: DataFrame, дополненный кортежами рёбер.
    """

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
    frame['predecessor_ids'], frame['connection_types'], frame['lags'] = predecessor_ids, connection_types, lags

    frame['edges'] = frame[['predecessor_ids', 'connection_types', 'lags']].apply(lambda row: list(zip(*row)), axis=1)
    return frame


def topsort_graph_df(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort works in topological order.

    Сортирует работы в топологическом порядке.

    Args:
        frame (pd.DataFrame): DataFrame of works.
            DataFrame работ.

    Returns:
        pd.DataFrame: Topologically sorted DataFrame.
            pd.DataFrame: топологически отсортированный DataFrame.
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


def build_work_graph(frame: pd.DataFrame, resource_names: list[str], work_estimator: WorkTimeEstimator) -> WorkGraph:
    """Construct a work graph from DataFrame data.

    Создаёт граф работ из данных DataFrame.

    Args:
        frame (pd.DataFrame): DataFrame with works and edges.
            DataFrame с работами и рёбрами.
        resource_names (list[str]): Names of resources.
            Названия ресурсов.
        work_estimator (WorkTimeEstimator): Estimator of work resources.
            Оценщик ресурсов для работ.

    Returns:
        WorkGraph: Built work graph.
            WorkGraph: построенный граф.
    """

    id_to_node = {}

    for _, row in frame.iterrows():
        if 'min_req' in frame.columns and 'max_req' in frame.columns and 'req_volume' in frame.columns:
            reqs = []
            for res_name in resource_names:
                if res_name in row['min_req'] and res_name in row['max_req']:
                    if 0 < row['min_req'][res_name] <= row['max_req'][res_name]:
                        reqs.append(
                            WorkerReq(
                                res_name,
                                Time(int(row['req_volume'][res_name])),
                                row['min_req'][res_name],
                                row['max_req'][res_name],
                            )
                        )
        else:
            reqs = work_estimator.find_work_resources(
                work_name=row['granular_name'],
                work_volume=float(row['volume']),
                measurement=row['measurement'],
            )
        is_service_unit = len(reqs) == 0

        zone_reqs = [ZoneReq(*v) for v in eval(row['required_statuses']).items()] \
            if 'required_statuses' in frame.columns else []

        description = row['description'] if 'description' in frame.columns else ''
        group = row['group'] if 'group' in frame.columns else 'main project'
        priority = row['priority'] if 'priority' in frame.columns else 1

        work_unit = WorkUnit(
            row['activity_id'],
            row['granular_name'],
            reqs,
            group=group,
            description=description,
            volume=row['volume'],
            volume_type=row['measurement'],
            is_service_unit=is_service_unit,
            display_name=row['activity_name_original'],
            zone_reqs=zone_reqs,
            priority=priority,
        )
        parents = [(id_to_node[p_id], lag, conn_type) for p_id, conn_type, lag in row.edges]
        node = GraphNode(work_unit, parents)
        id_to_node[row['activity_id']] = node

    all_nodes = [id_to_node[index] for index in list(set(id_to_node.keys()))]
    return WorkGraph.from_nodes(all_nodes)


def get_graph_contractors(path: str, contractor_name: str | None = 'ООО "***"') -> (
        list[Contractor], dict[str, float]):
    """Read contractor information from a CSV file.

    Считывает информацию о подрядчике из CSV-файла.

    Args:
        path (str): Path to the CSV with worker counts.
            Путь к CSV с количеством рабочих.
        contractor_name (str | None): Name of the contractor.
            Имя подрядчика.

    Returns:
        tuple[list[Contractor], dict[str, float]]: Contractors and workers capacity.
            tuple[list[Contractor], dict[str, float]]: подрядчики и их мощность по рабочим.
    """

    contractor_id = str(uuid4())
    workers_df = pd.read_csv(path, index_col='name')
    workers = {(name, 0): Worker(str(uuid4()), name, count * 2, contractor_id)
               for name, count in zip(workers_df.index, workers_df['count'])}
    workers_max_count = workers_df['count'].to_dict()
    contractors = [Contractor(contractor_id, contractor_name, list(workers_max_count.keys()), [], workers, {})]
    return contractors, workers_max_count
