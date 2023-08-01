from itertools import chain
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from sampo.schemas.contractor import Contractor, get_contractor_for_resources_schedule
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import DefaultWorkEstimator, WorkTimeEstimator
from sampo.userinput.parser.exception import InputDataException
from sampo.userinput.parser.general_build import add_graph_info, topsort_graph_df, build_work_graph, preprocess_graph_df
from sampo.userinput.parser.history import set_connections_info
from sampo.utilities.task_name import NameMapper


class CSVParser:

    @staticmethod
    def work_graph_and_contractors(project_info: str,
                                   history_data: str | None = None,
                                   contractor_info: str | None = None,
                                   contractor_types: list[int] | None = None,
                                   unique_work_names_mapper: NameMapper | None = None,
                                   work_resource_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                                   contractors_number: int = 1) \
            -> (WorkGraph, Contractor):
        """
        Gets a WorkGraph and Contractors from file .csv

        :param contractor_types:
        :param contractors_number: if we do not receive contractors, we need to know how many contractors the user wants,
        for a generation
        :param history_data: path to history data .csv file
        :param contractor_info: path to contractor info .csv file
        :param work_resource_estimator: work estimator that finds necessary resources, based on the history data
        :param project_info: path to .csv file
        :return: WorkGraph
        """

        if contractor_types is None:
            contractor_types = [25]

        graph_df = pd.read_csv(project_info, sep=';', header=0) if isinstance(project_info,
                                                                              str) else project_info
        # 1. work with tasks' connections. Preprocessing and breaking loops
        if history_data is None or (not(history_data is None) and 'predecessor_ids' in graph_df.columns):
            # 1.1. if we have info about predecessors in graph_df
            if 'predecessor_ids' not in graph_df.columns:
                raise InputDataException(
                    'you have neither history data about tasks nor tasks\' connection info in received .csv file.')
            works_info = preprocess_graph_df(graph_df)
        else:
            # 1.2. if we ought to restore predecessor info from history data
            history_df = pd.read_csv(history_data)
            graph_df = set_connections_info(graph_df, history_df)
            works_info = preprocess_graph_df(graph_df)

        # 2. gather resources and contractors based on work resource estimator or contractor info .csv file
        contractors = []
        resource_names = []
        works_info['activity_name_original'] = works_info.activity_name
        if unique_work_names_mapper:
            works_info.activity_name = works_info.activity_name.apply(lambda name: unique_work_names_mapper[name])

        if contractor_info is None:
            resources = [work_resource_estimator.find_work_resources(w[0], float(w[1]))
                         for w in works_info.loc[:, ['activity_name', 'volume']].to_numpy()]
            contractors = [get_contractor_for_resources_schedule(resources,
                                                                 contractor_capacity=contractor_types[i],
                                                                 contractor_id=str(i),
                                                                 contractor_name='Contractor' + ' ' + str(i + 1))
                           for i in range(contractors_number)]
        else:
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
            resource_names = contractor_df.columns[1:].to_list()
            if len(contractors) == 0 and isinstance(work_resource_estimator, DefaultWorkEstimator):
                raise InputDataException(
                    'you have neither info about contractors nor work resource estimator.'
                )
            resources = [work_resource_estimator.find_work_resources(w[0], float(w[1]), resource_names)
                         for w in works_info.loc[:, ['activity_name', 'volume']].to_numpy()]

        unique_res = list(set(chain(*[r.keys() for r in resources])))
        works_info.loc[:, unique_res] = DataFrame(resources).fillna(0)

        works_resources = add_graph_info(works_info)
        works_resources = topsort_graph_df(works_resources)
        work_graph = build_work_graph(works_resources, unique_res)

        # if we have no info about contractors or the user send an empty .csv file
        if len(contractors) == 0:
            contractors = [get_contractor_for_resources_schedule(resources,
                                                                 contractor_id=str(i),
                                                                 contractor_name='Contractor' + ' ' + str(i + 1))
                           for i in range(contractors_number)]

        return work_graph, contractors
