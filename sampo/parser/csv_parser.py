from itertools import chain

import pandas as pd
from pandas import DataFrame

from sampo.parser.general_build import add_graph_info, topsort_graph_df, build_work_graph, preprocess_graph_df
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time_estimator import DefaultWorkEstimator, WorkTimeEstimator
from sampo.utilities.task_name import NameMapper


class CSVParser:

    @staticmethod
    def work_graph(graph_info_file: str,
                   unique_work_names_mapper: NameMapper | None = None,
                   work_resource_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
            -> WorkGraph:
        """
        Gets a WorkGraph from file .csv

        :param unique_work_names_mapper:
        :param work_resource_estimator: work estimator, that find necessary resources, based on history data
        :param graph_info_file: Path to .csv file
        :return: WorkGraph
        """

        graph_df = pd.read_csv(graph_info_file, sep=';', header=0) if isinstance(graph_info_file,
                                                                                 str) else graph_info_file
        works_info = preprocess_graph_df(graph_df)

        works_info['activity_name_original'] = works_info.activity_name
        if unique_work_names_mapper:
            works_info.activity_name = works_info.activity_name.apply(lambda name: unique_work_names_mapper[name])

        resources = [work_resource_estimator.find_work_resources(w[0], float(w[1]))
                     for w in works_info.loc[:, ['activity_name', 'volume']].to_numpy()]

        unique_res = list(set(chain(*[r.keys() for r in resources])))
        works_info.loc[:, unique_res] = DataFrame(resources).fillna(0)

        works_resources = add_graph_info(works_info)
        works_resources = topsort_graph_df(works_resources)
        work_graph = build_work_graph(works_resources, unique_res)

        return work_graph
