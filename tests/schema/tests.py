import os
import shutil
from abc import ABC, abstractmethod
import unittest
from enum import Flag, auto
from typing import Any, Dict, Union, List, Optional

from pandas import DataFrame

from utilities.visualization.base import VisualizationMode
from tests.schema.reports import TestReport


class TestOutput(Flag):
    SaveCsv = auto()
    CreateFig = auto()


class KsgTest(ABC, unittest.TestCase):
    _output: TestOutput
    _fig_visualization: VisualizationMode

    _report_folder: str = './reports/'

    _reports: Union[Dict[str, DataFrame], Dict[str, Any]] = dict()
    _failures: Dict[str, List[TestReport]] = dict()

    @property
    def errors(self):
        return {k: v for k, v in [(n, [i for i in l if i.is_error]) for n, l in self._failures.items()] if v}

    @property
    def warnings(self):
        return {k: v for k, v in [(n, [i for i in l if i.is_warning]) for n, l in self._failures.items()] if v}

    @abstractmethod
    def save_text_reports(self, report_folder: str):
        ...

    @abstractmethod
    def generate_fig_reports(self, visualization: VisualizationMode, report_folder: Optional[str] = None):
        ...

    def setup_test(self, output: TestOutput, visualization: Optional[VisualizationMode] = VisualizationMode.NoFig):
        self._output = output
        self._fig_visualization = visualization

    def tearDown(self):
        test_name = self.__class__.__name__
        report_folder = os.path.join(self._report_folder, test_name)
        if TestOutput.SaveCsv in self._output:
            self._refresh_csv_reports(report_folder)
        if TestOutput.CreateFig in self._output:
            self.generate_fig_reports(self._fig_visualization, report_folder)

    def report_failure(self, report: TestReport):
        if report.test_name in self._failures:
            self._failures[report.test_name].append(report)
        else:
            self._failures[report.test_name] = [report]

    def _refresh_csv_reports(self, report_folder):
        # remove old reports, if exist
        if os.path.isdir(report_folder):
            shutil.rmtree(report_folder)
        elif not os.path.isdir(self._report_folder):
            os.mkdir(self._report_folder)
        os.mkdir(report_folder)
        self.save_text_reports(report_folder)
