import unittest
from itertools import product

from examples.launch_scheduling_algorithms import prepare_work_graph, schedule_works
from launch.schemas.configuration.graph import GraphConfiguration
from launch.schemas.configuration.scheduling import SchedulingInputConfiguration, SchedulingInnerConfiguration, \
    SchedulingOutputConfiguration, SchedulingConfiguration


class TestScheduling(unittest.TestCase):
    def test_real_blocks(self):
        resources_path = '../resources/'

        real_data_names_times = ('cluster_25', 223), ('water_pipe', 28), ('power_line', 84), ('light_mast', 42)
        model_types = 'baps', 'bapl'
        algorithm_types = 'heft_add_between', 'heft_add_end', 'topological', 'genetic'

        options = product(real_data_names_times, model_types, algorithm_types)

        for (block_name, end_time), model_type, algorithm_type in options:
            start_date = '2019-02-22'
            plot_title = f'{block_name}_'
            filepath = resources_path + f'pickle_dumps/brksg/baps_lag_{block_name}.pickle'
            info_dict = resources_path + f'brksg/{block_name}/works_info.csv'

            graph_config = GraphConfiguration.read_graph(graph_info_filepath=filepath,
                                                         use_graph_lag_optimization=model_type == 'baps')

            wg, contractors = prepare_work_graph(*graph_config.dump_config_params())

            scheduling_input = SchedulingInputConfiguration(work_graph=wg, contractors=contractors, ksg_info=info_dict)
            scheduling_inner = SchedulingInnerConfiguration(algorithm_type=algorithm_type,
                                                            start=start_date,
                                                            validate_schedule=True)
            scheduling_output = SchedulingOutputConfiguration(title=plot_title,
                                                              output_folder='output/',
                                                              save_to_xer=False,
                                                              save_to_csv=False,
                                                              save_to_json=False)

            try:
                schedule_works(*SchedulingConfiguration(scheduling_input, scheduling_inner, scheduling_output)
                               .dump_config_params())
            except Exception as ex:
                self.assertEqual(True, False, ex)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
