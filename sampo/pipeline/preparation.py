import pandas as pd

from sampo.schemas import WorkGraph, Contractor, WorkTimeEstimator
from sampo.schemas.structure_estimator import StructureEstimator
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.userinput import CSVParser
from sampo.utilities.name_mapper import NameMapper


class PreparationPipeline:

    def __init__(self,
                 wg: str | pd.DataFrame | WorkGraph = None,
                 history: str | pd.DataFrame | None = None,
                 contractors: str | pd.DataFrame | list[Contractor] | None = None):
        self._wg = wg
        self._contractors = contractors
        self._history = history if history is not None else pd.DataFrame(columns=['marker_for_glue', 'work_name',
                                                                                  'first_day', 'last_day',
                                                                                  'upper_works', 'work_name_clear_old',
                                                                                  'smr_name', 'work_name_clear',
                                                                                  'granular_smr_name'])
        self._fix_edges_with_history = False
        self._fill_edges_with_history = False
        self._sep_wg = ';'
        self._sep_history = ';'
        self._name_mapper = None
        self._work_estimator = DefaultWorkEstimator()
        self._structure_estimator = None

    def wg(self, wg: str | pd.DataFrame | WorkGraph):
        self._wg = wg
        return self

    def history(self, history: str | pd.DataFrame):
        self._history = history
        return self

    def contractors(self, contractors: str | pd.DataFrame | list[Contractor]):
        self._contractors = contractors
        return self

    def fix_edges_with_history(self) -> 'PreparationPipeline':
        self._fix_edges_with_history = True
        return self

    def fill_edges_with_history(self) -> 'PreparationPipeline':
        self._fill_edges_with_history = True
        return self

    def sep_wg(self, sep: str) -> 'PreparationPipeline':
        self._sep_wg = sep
        return self

    def sep_history(self, sep: str) -> 'PreparationPipeline':
        self._sep_history = sep
        return self

    def name_mapper(self, name_mapper: NameMapper) -> 'PreparationPipeline':
        self._name_mapper = name_mapper
        return self

    def work_estimator(self, work_estimator: WorkTimeEstimator) -> 'PreparationPipeline':
        self._work_estimator = work_estimator
        return self

    def structure_estimator(self, structure_estimator: StructureEstimator) -> 'PreparationPipeline':
        self._structure_estimator = structure_estimator
        return self

    def _prepare_works_info(self) -> pd.DataFrame:
        wg = self._wg.to_frame(save_req=True) if isinstance(self._wg, WorkGraph) else self._wg
        return CSVParser.read_graph_info(project_info=wg,
                                         history_data=self._history,
                                         sep_wg=self._sep_wg,
                                         sep_history=self._sep_history,
                                         name_mapper=self._name_mapper,
                                         all_connections=not self._fix_edges_with_history and
                                                         not self._fill_edges_with_history,
                                         change_connections_info=self._fix_edges_with_history)

    def build_wg(self) -> WorkGraph:
        wg = CSVParser.work_graph(
            works_info=self._prepare_works_info(),
            name_mapper=self._name_mapper,
            work_resource_estimator=self._work_estimator
        )

        if self._structure_estimator is not None:
            wg = self._structure_estimator.restruct(wg)

        return wg

    def build_wg_and_contractors(self) -> tuple[WorkGraph, list[Contractor]]:
        return CSVParser.work_graph_and_contractors(
            works_info=self._prepare_works_info(),
            name_mapper=self._name_mapper,
            contractor_info=self._contractors,
            work_resource_estimator=self._work_estimator
        )
