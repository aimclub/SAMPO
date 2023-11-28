import math
from itertools import chain
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from sampo.generator.environment import ContractorGenerationMethod, get_contractor_by_wg
from sampo.schemas.contractor import Contractor, get_contractor_for_resources_schedule
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import DefaultWorkEstimator, WorkTimeEstimator
from sampo.userinput.parser.exception import InputDataException
from sampo.userinput.parser.general_build import add_graph_info, topsort_graph_df, build_work_graph, \
    preprocess_graph_df, break_loops_in_input_graph
from sampo.userinput.parser.history import set_connections_info
from sampo.utilities.name_mapper import NameMapper


class CSVParser:

    @staticmethod
    def read_graph_info(project_info: str | pd.DataFrame,   # TODO Fix mutating input data
                        history_data: str | pd.DataFrame,
                        sep: str,
                        full_connections: bool = False,
                        change_base_on_history: bool = False) -> pd.DataFrame:
        """
        Read the input data about work graph and preprocess it.

        Schema of WorkGraph .csv file:
            mandatory fields:
                activity_id: str - Id of the current task,
                activity_name: str - Name of the current task,
                measurement: str - Measure of the size of the current task (e.g., km, pcs, lit),
                volume: float - Volume of the current task
            optional fields:
                granular_name: str - Task name as in the document,
                predecessor_ids: list[str] - Ids of predecessors of the current task,
                connection_types: list[str] - Types of links between the current task and its predecessors,
                lags: float - Time lags,
                min_req: dict[str, float] - A dictionary containing the minimum amount of each resource
                                            that is required to perform the current task
                max_req: dict[str, float] - A dictionary containing the maximum amount of each resource
                                            that is required to perform the current task
                description: str - A task description
                required_statuses: dict[str, int] - A dictionary containing the zone statuses required
                                                    to perform the current task

        !NOTE! that length of predecessor_ids, connection_types and lags in each cell should be equal
        for the correct predecessor resolution

        Schema of history .csv file (optional data):
            mandatory fields:
                granular_smr_name: str - Task name as in the document,
                first_day: str - Date of commencement of the work,
                last_day: str - Date of completion
                upper_works: list[str] - Names of predecessors of the current task

        ATTENTION!
            1) If you send WorkGraph .csv file without data about connections between tasks, you need to provide .csv
            history file - the SAMPO will be able to reconstruct the connections between tasks based on historical data.
            2) If you send WorkGraph .csv file with column 'predecessor_ids', 'lags' etc. and there is no info in these
            columns, so framework repair the info from history data
            3) If you send WorkGraph .csv or HistoryData file path, use the same separating character in
            work_info.csv as in history_data.csv and vice versa.

        :param project_info: path to the works' info file
        :param history_data: path to the history data of connection file
        :param full_connections: does the project information contain full details of the works?
        :param change_base_on_history: whether it is necessary to change project information based on connection history data?
        :param sep: separating character. It's mandatory, if you send the WorkGraph .csv or HistoryData .csv file path
        :return: preprocessed info about works
        """
        graph_df = pd.read_csv(project_info, sep=sep, header=0) if isinstance(project_info,
                                                                              str) else project_info.copy()
        history_df = pd.read_csv(history_data, sep=sep) if isinstance(history_data, str) else history_data.copy()

        if 'predecessor_ids' not in graph_df.columns and history_data.shape[0] == 0:
            raise InputDataException(
                'you have neither history data about tasks nor tasks\' connection info in received .csv file.')

        if 'predecessor_ids' not in graph_df.columns:
            # if we ought to restore predecessor info from history data
            temp_lst = [math.nan] * graph_df.shape[0]
            for col in ['predecessor_ids', 'connection_types', 'lags']:
                graph_df[col] = temp_lst
            graph_df = preprocess_graph_df(graph_df)
            works_info = set_connections_info(graph_df, history_df)
        else:
            graph_df = preprocess_graph_df(graph_df)
            if full_connections or change_base_on_history:
                works_info = set_connections_info(graph_df, history_df, change_connections_info=True)
            else:
                works_info = set_connections_info(graph_df, history_df, expert_connections_info=True)

        return break_loops_in_input_graph(works_info)

    @staticmethod
    def work_graph_and_contractors(works_info: pd.DataFrame,
                                   contractor_info: str | list[Contractor] | tuple[ContractorGenerationMethod, int]
                                                                              = (ContractorGenerationMethod.AVG, 1),
                                   contractor_types: list[int] | None = None,
                                   unique_work_names_mapper: NameMapper | None = None,
                                   work_resource_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
            -> (WorkGraph, Contractor):
        """
        Gets a info about WorkGraph and Contractors from file .csv.

        Schema of Contractors .csv file (optional data):
            mandatory fields:
                contractor_id: str - Id of the current contractor,
                name: str - Contractor name as in the document
            optional fields:
                {names of resources}: float - each resource is a separate column

        ATTENTION!
            1) If you do not provide work resource estimator, framework uses built-in estimator


        :param works_info: dataFrame that contains preprocessed info about work graph structure
        :param contractor_types:
        :param contractors_number: if we do not receive contractors, we need to know how many contractors the user wants,
        for a generation
        :param contractor_info: path to contractor info .csv file or list of Contractors
        :param work_resource_estimator: work estimator that finds necessary resources, based on the history data
        :return: WorkGraph and list of Contractors
        """

        if contractor_types is None:
            contractor_types = [25]

        # gather resources and contractors based on work resource estimator or contractor info .csv file
        contractors = []
        works_info['activity_name_original'] = works_info.activity_name
        if unique_work_names_mapper:
            works_info.activity_name = works_info.activity_name.apply(lambda name: unique_work_names_mapper[name])

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

        resources = [dict((worker_req.kind, int(worker_req.volume))
                          for worker_req in work_resource_estimator.find_work_resources(w[0], float(w[1])))
                     for w in works_info.loc[:, ['activity_name', 'volume']].to_numpy()]

        unique_res = list(set(chain(*[r.keys() for r in resources])))
        # works_info.loc[:, unique_res] = DataFrame(resources).fillna(0)

        works_resources = add_graph_info(works_info)
        works_resources = topsort_graph_df(works_resources)
        work_graph = build_work_graph(works_resources, unique_res, work_resource_estimator)

        # if we have no info about contractors or the user send an empty .csv file
        if len(contractors) == 0:
            generation_method, contractors_number = contractor_info
            contractors = [get_contractor_by_wg(work_graph,
                                                method=generation_method,
                                                contractor_id=str(i),
                                                contractor_name='Contractor' + ' ' + str(i + 1))
                           for i in range(contractors_number)]

        return work_graph, contractors
