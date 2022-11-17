import os
import unittest
from datetime import datetime
from itertools import product
from operator import itemgetter
from time import time
from typing import Tuple, Optional, Union, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from pandas import concat, DataFrame

from examples.launch_scheduling_algorithms import prepare_work_graph
from external.estimate_time import WorkResourceEstimator, WorkTimeEstimator
from metrics.resources_in_time.base import max_time_schedule
from metrics.resources_in_time.service import ResourceOptimizationType, apply_resource_optimization
from scheduler.base import SchedulerType
from scheduler.generate import get_scheduler_ctor
from scheduler.heft.base import HEFTScheduler
from schemas.contractor import Contractor, get_worker_contractor_pool, ContractorType
from schemas.time import Time
from utilities import get_inverse_task_name_mapping, get_task_name_unique_mapping
from utilities.visualization.base import VisualizationMode, visualize
from launch.schemas.configuration.graph import GraphConfiguration
from launch.schemas.configuration.scheduling import SchedulingConfiguration, SchedulingInputConfiguration, \
    SchedulingInnerConfiguration, SchedulingOutputConfiguration
from tests.schema.exceptions import TestException
from tests.schema.reports import ComparisonTestReport
from tests.schema.tests import TestOutput, KsgTest

MAX_EXPERIMENT_ITERATIONS: int = 12
OUTPUT: TestOutput = TestOutput.SaveCsv | TestOutput.CreateFig


def label_keys(d: Union[dict, DataFrame], label: str, suffix: bool = False, sep: str = '_') -> dict:
    return {f'{key}{sep}{label}' if suffix else f'{label}{sep}{key}': value for key, value in d.items()} \
        if isinstance(d, dict) \
        else d.rename({col: f'{col}{sep}{label}' if suffix else f'{label}{sep}{col}' for col in d.columns}, axis=1)


def setup_scheduling_params(algorithm_type, block_name, contractor, generate_resources, lag_optimization,
                            resources_path, work_time_estimator,
                            res_opt: Optional[Tuple[float, ResourceOptimizationType]] = None):
    unique_names_dict_path = f'{resources_path}unique_tasks_dict.csv'
    start_date = '2019-02-22'
    plot_title = f'real_block_{block_name}'
    filepath = f'{resources_path}pickle_dumps/brksg/baps_lag_{block_name}.pickle'
    info_dict = f'{resources_path}brksg/{block_name}/works_info.csv'
    ivanov_model_pickle = '../src/external/lib/estimate_time/data/trained_models_M21_productivity.pickle'
    res_estimator = None
    name_mapper = get_task_name_unique_mapping(unique_names_dict_path)
    inv_mapper = get_inverse_task_name_mapping(unique_names_dict_path)
    if generate_resources:
        res_estimator = WorkResourceEstimator(ivanov_model_pickle)
        graph_config = GraphConfiguration \
            .read_graph_generate_resources(graph_info=info_dict,
                                           use_graph_lag_optimization=lag_optimization,
                                           work_resource_estimator=res_estimator,
                                           contractor_type=contractor,
                                           unique_work_names_mapper=name_mapper)
    else:
        graph_config = GraphConfiguration.read_graph(graph_info_filepath=filepath,
                                                     use_graph_lag_optimization=lag_optimization)
    wg, contractors = prepare_work_graph(*graph_config.dump_config_params())
    n_vertices = wg.vertex_count
    scheduling_input = SchedulingInputConfiguration(work_graph=wg, contractors=contractors, ksg_info=info_dict)
    scheduling_inner = SchedulingInnerConfiguration(algorithm_type=algorithm_type,
                                                    start=start_date,
                                                    validate_schedule=False) \
        .with_inverse_name_mapper(inverse_name_mapper=inv_mapper)
    if work_time_estimator:
        time_estimator = WorkTimeEstimator(res_estimator or WorkResourceEstimator(ivanov_model_pickle))
        scheduling_inner.with_work_time_estimator(work_time_estimator=time_estimator, use_idle_estimator=False)
    if res_opt:
        scheduling_inner.with_deadline_resource_optimization(*res_opt)
    scheduling_output = SchedulingOutputConfiguration(title=f'{plot_title}_{n_vertices}_tasks',
                                                      output_folder='output/',
                                                      save_to_xer=False,
                                                      save_to_csv=False,
                                                      save_to_json=False)
    scheduling_config = SchedulingConfiguration(scheduling_input, scheduling_inner, scheduling_output)
    return scheduling_config


def apply_scheduling(algorithm_type: SchedulerType, scheduling_config: SchedulingConfiguration) \
        -> Union[Tuple[Contractor, Time], Time]:
    if scheduling_config.inner_configuration.work_time_estimator:
        scheduling_config.inner_configuration.work_time_estimator.set_mode(
            scheduling_config.inner_configuration.use_idle_estimator
        )
    agents = get_worker_contractor_pool(scheduling_config.input_configuration.contractors)
    if scheduling_config.inner_configuration.resource_optimization \
            and scheduling_config.inner_configuration.deadline:
        return apply_resource_optimization(
            scheduler=HEFTScheduler(scheduling_config.inner_configuration.work_time_estimator),
            work_graph=scheduling_config.input_configuration.work_graph,
            deadline=scheduling_config.inner_configuration.deadline,
            agents_from_manual_input=agents,
            optimization_type=scheduling_config.inner_configuration.resource_optimization
        )
    else:
        scheduler = get_scheduler_ctor(algorithm_type)(scheduling_config.inner_configuration.work_time_estimator)
        sch = scheduler.schedule(wg=scheduling_config.input_configuration.work_graph,
                                 contractors=scheduling_config.input_configuration.contractors,
                                 validate_schedule=scheduling_config.inner_configuration
                                 .validate_schedule)
        return max_time_schedule(sch[0] if isinstance(sch, tuple) else sch)


def schedule(algorithm_type, block_name, contractor, generate_resources, lag_optimization, resources_path,
             resource_optimization, work_time_estimator):
    scheduling_config = setup_scheduling_params(algorithm_type, block_name, contractor, generate_resources,
                                                lag_optimization, resources_path, work_time_estimator,
                                                res_opt=resource_optimization)
    reference_configuration_time: Time = apply_scheduling(algorithm_type, scheduling_config)
    return reference_configuration_time


def flatten_dict(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return tuple(np.array(l) for l in zip(*list(sorted(d.items(), key=itemgetter(0)))))


class TestResourceOptimization(KsgTest):
    def setUp(self):
        self.setup_test(OUTPUT, VisualizationMode.SaveFig)

    def test_real_blocks(self):
        """
        Tests, whether the new written gradient resource optimization performs not worse than binary search based one
        """
        resources_path = '../resources/'

        test_start = datetime.fromtimestamp(time())
        test_name = 'test_real_blocks'

        real_data_names_times = 'cluster_25', 'electroline'
        algorithm_type = SchedulerType.HEFTAddBetween
        lag_optimization_options = False,
        generate_resource_and_contractor_options = [(True, ct) for ct in list(ContractorType)]  # + [(False, None)]
        use_work_time_estimator = False,  # True, False

        options = product(real_data_names_times, lag_optimization_options,
                          generate_resource_and_contractor_options, use_work_time_estimator)

        result = DataFrame()

        for experiment, (block_name, lag_optimization, (generate_resources, contractor),
                         work_time_estimator) in enumerate(options):
            exp_info = {'info_test_name': test_name,
                        'info_start': test_start,
                        'info_experiment_n': experiment,
                        'info_ksg_block': block_name,
                        'info_scheduling_algorithm': algorithm_type.name,
                        'info_lag_optimization': 'baps' if lag_optimization else 'bapl',
                        'info_resource_generation': generate_resources,
                        'info_work_time_estimator': work_time_estimator,
                        'info_contractor_type': contractor and contractor.name}

            # shortcut for convenient optimization
            def ctrl_sch(optimization=Union[ResourceOptimizationType, None], deadline: Optional[Time] = None) \
                    -> Union[Time, Tuple[Contractor, Time]]:
                return schedule(algorithm_type, block_name, contractor, generate_resources,
                                lag_optimization, resources_path, (deadline, optimization), work_time_estimator) \
                    if optimization and deadline \
                    else schedule(algorithm_type, block_name, contractor, generate_resources,
                                  lag_optimization, resources_path, None, work_time_estimator)

            reference_configuration_time = ctrl_sch(None)
            contractor_lower_bounds, _ = ctrl_sch(ResourceOptimizationType.BinarySearch,
                                                  reference_configuration_time * 3)
            cols, agents_lower_bound = flatten_dict(get_worker_contractor_pool(contractor_lower_bounds))

            i = 0
            deadline = reference_configuration_time
            curr_contractor = agents_lower_bound + 1
            while (curr_contractor > agents_lower_bound).any() and i < MAX_EXPERIMENT_ITERATIONS:
                iter_info = dict({'iter_info_deadline': deadline}, **exp_info)

                bin_contractor, bin_len = ctrl_sch(ResourceOptimizationType.BinarySearch, deadline)
                grad_contractor, grad_len = ctrl_sch(ResourceOptimizationType.NewtonCG, deadline)

                ((bin_agent_names, bin_agents),
                 (grad_agent_names, grad_agents)) = (flatten_dict(get_worker_contractor_pool(c))
                                                     for c in (bin_contractor, grad_contractor))

                report = ComparisonTestReport(test_name, test_start, iter_info,
                                              list(bin_agents),
                                              list(grad_agents))
                try:
                    self.assertTrue((bin_agent_names == grad_agent_names).all())
                except AssertionError:
                    self.report_failure(report.with_exception(TestException.error(
                        'Resource optimizations have produced different set of resources.'
                        f'\nbin agent names:\t{list(bin_agent_names)}'
                        f'\ngrad agent names: \t{list(grad_agent_names)}'
                    )))

                if grad_agents.sum() > bin_agents.sum():
                    self.report_failure(report.with_exception(TestException.error(
                        'Gradient optimization returned more expensive resources in experiment'
                    )))
                    # self.report_failure(report.with_exception(TestException.warning(
                    #     'Gradient optimization returned higher resources in experiment'
                    # )))
                elif (grad_agents > bin_agents).any():
                    self.report_failure(report.with_exception(TestException.warning(
                        'Gradient optimization returned higher resources in experiment'
                    )))
                curr_contractor = grad_agents

                current_results = (DataFrame(
                    dict({'optimizers': ((ResourceOptimizationType.BinarySearch.name,
                                          ResourceOptimizationType.NewtonCG.name),),
                          'agents': (cols,),
                          f'{ResourceOptimizationType.BinarySearch.name}_schedule_length': bin_len,
                          f'{ResourceOptimizationType.NewtonCG.name}_schedule_length': grad_len,
                          f'{ResourceOptimizationType.BinarySearch.name}_agents': (bin_agents,),
                          f'{ResourceOptimizationType.NewtonCG.name}_agents': (grad_agents,),
                          f'residuals': (bin_agents - grad_agents,)},
                         **iter_info), index=[0]))

                result = concat((result, current_results))

                i += 1
                deadline += 1

            if experiment % 5 == 0 and experiment > 0:
                print(f'Experiment\t{experiment}, ok')
        result.index = range(result.shape[0])
        self._reports[test_name] = result

    def save_text_reports(self, report_folder: str):
        """
        First, deletes the former csv reports of this test class. Then, creates and saves reports on:
            1. Each experiment's optimization data with binary and gradient features and residuals;
            2. All the failures in up to two files:
                1. Warnings, if any;
                2. Error, if the test has finished with error.
        """

        def f_name(name, ext='csv', suffix=''):
            return os.path.join(report_folder, name + (suffix and f'_{suffix}') + f'.{ext}')

        for n, v in self._reports.items():
            v.to_csv(f_name(n), encoding='utf-8')

        for evt, evt_n in ((self.warnings, 'wrn'), (self.errors, 'err')):
            if evt:
                for n, v in evt.items():
                    with open(f_name(n, 'json', evt_n), 'w', encoding='utf-8') as write_file:
                        write_file.write(f'{{"{evt_n}": [{",".join([w.dumps() for w in v])}]}}')

    def generate_fig_reports(self, visualization: VisualizationMode, report_folder: Optional[str] = None):
        """
        Graph of loss, 2 lines:
          1 - Number of failed resources
          2 - Total difference of failed resources' quantity
        ---
        Graph of target function, 2 lines:
          1 - Weighted sum of resources for binary optimization
          2 - -//- for gradient optimization
        ---
        Pie chart with succeed/failed fraction
        :param visualization: visualization mode
        :param report_folder: path to report folder, if needed to save
        """

        def loss_fig(ax, ind, data):
            failed_res = data.loc[ind, 'residuals'].apply(lambda r: (r < 0).sum())
            total_failed = data.loc[ind, 'residuals'].apply(lambda r: sum([a for a in r if a < 0]))
            lfr = len(failed_res)
            assert lfr == len(total_failed)
            x = np.linspace(1, lfr, lfr)
            ax.set_title('Loss')
            ax.plot(x, failed_res, x, total_failed)
            ax.legend(('Number of failed resources', 'Total loss for failed resources'))

        def cost_fig(ax, ind, data):
            ax.set_title('Cost')
            x = np.linspace(1, len(ind), len(ind))
            cols = list(data.loc[0, 'optimizers'])
            for col in cols:
                line = data.loc[ind, f'{col}_agents'].apply(sum)
                ax.plot(x, line)
            ax.legend(cols)

            # cols = [f'{o}_agents' for o in data.loc[0, 'optimizers']]
            # lines = list(zip(*data.loc[ind, cols].apply(lambda x: [x[col].sum() for col in cols], axis=1)))
            # x = np.linspace(1, len(lines[0]), len(lines[0]))
            # for line in lines:
            #     ax.plot(x, line)
            # ax.legend(list(data.loc[0, 'optimizers']))

        def piechart(ax, ind, name, test_name):
            wrn, err = ((len([True for w in evt[test_name] if w.info['ksg_block'] == name])
                         if test_name in evt
                         else 0) for evt in (self.warnings, self.errors))
            ax.set_title('Exceptions')
            s = [len(ind) - wrn, wrn - err, err]
            print(s)
            ax.pie(s, labels=['OK', 'Warning', 'Error'], autopct='%1.1f%%')
            ax.axis('equal')

        for test, result in self._reports.items():
            grouped = result.groupby(by='info_ksg_block')
            for gr, lbl in grouped.groups.items():
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))
                fig.suptitle(gr, fontsize=16)

                loss_fig(ax1, lbl, result)
                cost_fig(ax2, lbl, result)
                piechart(ax3, lbl, gr, test)
                visualize(fig, visualization, report_folder and os.path.join(report_folder, f'{test}_{gr}.png'))


if __name__ == '__main__':
    unittest.main()
