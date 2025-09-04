"""Parsing utilities for building graphs and contractors from CSV files.

Инструменты разбора CSV-файлов для построения графов и подрядчиков.
"""

import math
from itertools import chain
from uuid import uuid4

import pandas as pd

from sampo.generator.environment import ContractorGenerationMethod, get_contractor_by_wg
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import DefaultWorkEstimator, WorkTimeEstimator
from sampo.userinput.parser.exception import InputDataException
from sampo.userinput.parser.general_build import add_graph_info, topsort_graph_df, build_work_graph, \
    preprocess_graph_df, break_loops_in_input_graph
from sampo.userinput.parser.history import set_connections_info
from sampo.utilities.name_mapper import NameMapper


class CSVParser:
    """Parser for reading work graphs and contractor data from CSV.

    Парсер для чтения графов работ и данных подрядчиков из CSV.
    """

    @staticmethod
    def read_graph_info(project_info: str | pd.DataFrame,
                        history_data: str | pd.DataFrame,
                        sep_wg: str = ';',
                        sep_history: str = ';',
                        name_mapper: NameMapper | None = None,
                        all_connections: bool = False,
                        change_connections_info: bool = False) -> pd.DataFrame:
        """Read and preprocess work graph and history data.

        Читает и предварительно обрабатывает граф работ и данные истории.

        The function parses two CSV sources: a work graph description and an
        optional history file used to restore missing connections.

        Функция разбирает два CSV-источника: описание графа работ и необязательный
        файл истории, применяемый для восстановления отсутствующих связей.

        WorkGraph CSV columns:
            Required:
                activity_id, activity_name, measurement, volume
            Optional:
                granular_name, predecessor_ids, connection_types, lags,
                min_req, max_req, description, required_statuses

        История CSV (необязательна) должна содержать столбцы:
            granular_name, first_day, last_day, upper_works

        Notes / Примечания:
            * Lengths of ``predecessor_ids``, ``connection_types`` and ``lags``
              must match.
            * If the work graph lacks connections, historical data is required.
              Если в графе нет связей, необходимо предоставить историю.
            * When ``granular_name`` is absent and ``name_mapper`` is not given,
              ``activity_name`` must match the historical name.
              Если нет ``granular_name`` и ``name_mapper``,
              ``activity_name`` должно совпадать с историческим.

        Args:
            project_info (str | pd.DataFrame): Path or dataframe with work graph data.
                Путь или DataFrame с данными графа работ.
            history_data (str | pd.DataFrame): Path or dataframe with history data.
                Путь или DataFrame с историческими данными.
            sep_wg (str): Separator for work graph CSV.
                Разделитель в CSV графа работ.
            sep_history (str): Separator for history CSV.
                Разделитель в CSV истории.
            name_mapper (NameMapper | None): Mapper from ``activity_name`` to ``granular_name``.
                Сопоставитель ``activity_name`` и ``granular_name``.
            all_connections (bool): Replace existing connections entirely.
                Полностью заменять существующие связи.
            change_connections_info (bool): Modify connection info using history.
                Изменять информацию о связях на основе истории.

        Returns:
            pd.DataFrame: Preprocessed work information.
                pd.DataFrame: предварительно обработанные данные о работах.
        """
        graph_df = pd.read_csv(project_info, sep=sep_wg, header=0) if isinstance(project_info,
                                                                                 str) else project_info.copy()
        history_df = pd.read_csv(history_data, sep=sep_history) if isinstance(history_data,
                                                                              str) else history_data.copy()

        if 'predecessor_ids' not in graph_df.columns and history_df.shape[0] == 0:
            raise InputDataException(
                'you have neither history data about tasks nor tasks\' connection info in received .csv file.')

        graph_df = preprocess_graph_df(graph_df, name_mapper)
        id2ind = {graph_df.loc[i, 'activity_id']: i for i in range(len(graph_df.index))}
        works_info = set_connections_info(graph_df, history_df, mapper=name_mapper,
                                          all_connections=all_connections,
                                          change_connections_info=change_connections_info, id2ind=id2ind)

        return break_loops_in_input_graph(works_info)

    @staticmethod
    def work_graph(works_info: pd.DataFrame,
                   name_mapper: NameMapper | None = None,
                   work_resource_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> WorkGraph:
        """Build a work graph from preprocessed information.

        Строит граф работ из предварительно обработанных данных.

        Args:
            works_info (pd.DataFrame): DataFrame with prepared work graph info.
                DataFrame с подготовленной информацией о графе работ.
            name_mapper (NameMapper | None): Mapper of activity names.
                Сопоставитель названий работ.
            work_resource_estimator (WorkTimeEstimator): Estimator of required resources.
                Оценщик необходимых ресурсов.

        Returns:
            WorkGraph: Constructed work graph.
                WorkGraph: построенный граф работ.
        """

        works_info['activity_name_original'] = works_info.activity_name
        if name_mapper:
            works_info.activity_name = works_info.activity_name.apply(lambda name: name_mapper[name])

        resources = [dict((worker_req.kind, int(worker_req.volume))
                          for worker_req in work_resource_estimator.find_work_resources(work_name=w[0],
                                                                                        work_volume=float(w[1]),
                                                                                        measurement=w[2]))
                     for w in works_info.loc[:, ['granular_name', 'volume', 'measurement']].to_numpy()]

        unique_res = list(set(chain(*[r.keys() for r in resources])))

        works_resources = add_graph_info(works_info)
        works_resources = topsort_graph_df(works_resources)
        work_graph = build_work_graph(works_resources, unique_res, work_resource_estimator)

        return work_graph

    @staticmethod
    def work_graph_and_contractors(works_info: pd.DataFrame,
                                   contractor_info: str | list[Contractor] | tuple[ContractorGenerationMethod, int]
                                   = (ContractorGenerationMethod.AVG, 1),
                                   contractor_types: list[int] | None = None,
                                   name_mapper: NameMapper | None = None,
                                   work_resource_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
            -> tuple[WorkGraph, list[Contractor]]:
        """Build a work graph and obtain contractors.

        Строит граф работ и получает список подрядчиков.

        Args:
            works_info (pd.DataFrame): Preprocessed work graph info.
                Предварительно обработанные данные графа работ.
            contractor_info (str | list[Contractor] |
                tuple[ContractorGenerationMethod, int]): Path to contractor CSV,
                predefined contractors, or generation parameters.
                Путь к CSV, список подрядчиков или параметры генерации.
            contractor_types (list[int] | None): Types of contractors to generate.
                Типы подрядчиков для генерации.
            name_mapper (NameMapper | None): Mapper of activity names.
                Сопоставитель названий работ.
            work_resource_estimator (WorkTimeEstimator): Estimator of required resources.
                Оценщик необходимых ресурсов.

        Returns:
            tuple[WorkGraph, list[Contractor]]: Built work graph and contractors list.
                tuple[WorkGraph, list[Contractor]]: построенный граф и список подрядчиков.
        """

        if contractor_types is None:
            contractor_types = [25]

        # gather resources and contractors based on work resource estimator or contractor info .csv file
        contractors = []

        if isinstance(contractor_info, list):
            contractors = contractor_info
        elif isinstance(contractor_info, str):
            # if contractor info is given or contractor info and work resource estimator are received simultaneously
            contractor_df = pd.read_csv(contractor_info, sep=';', header=0) if isinstance(contractor_info,
                                                                                          str) else contractor_info
            for index, row in contractor_df.iterrows():
                specified_id = 'contractor_id' in contractor_df.columns
                column_lag = 2 if specified_id else 1
                contractor_id = row['contractor_id'] if specified_id else str(uuid4())
                contractors.append(
                    Contractor(id=contractor_id,
                               name=row['name'],
                               workers={item[0]: Worker(str(ind), str(item[0]), int(item[1])) for ind, item in
                                        enumerate(row[column_lag:].items())},
                               equipments=dict())
                )

        work_graph = CSVParser.work_graph(works_info, name_mapper, work_resource_estimator)

        # if we have no info about contractors or the user send an empty .csv file
        if len(contractors) == 0:
            generation_method, contractors_number = contractor_info
            contractors = [get_contractor_by_wg(work_graph,
                                                method=generation_method,
                                                contractor_id=str(i),
                                                contractor_name='Contractor' + ' ' + str(i + 1))
                           for i in range(contractors_number)]

        return work_graph, contractors
