from sampo.generator import SimpleSynthetic
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline.default import DefaultInputPipeline
from sampo.scheduler import GeneticScheduler
from sampo.utilities.visualization import VisualizationMode

if __name__ == '__main__':

    # Set up scheduling algorithm and project's start date
    start_date = "2023-01-01"

    # Set up visualization mode (ShowFig or SaveFig) and the gant chart file's name (if SaveFig mode is chosen)
    visualization_mode = VisualizationMode.ShowFig
    gant_chart_filename = './output/synth_schedule_gant_chart.png'

    # Generate synthetic graph with material requirements for
    # number of unique works names and number of unique resources
    ss = SimpleSynthetic(rand=31)
    wg = ss.work_graph(top_border=200)
    wg = ss.set_materials_for_wg(wg)
    landscape = ss.synthetic_landscape(wg)

    # Be careful with the high number of generations and size of population
    # It can lead to a long time of the scheduling process because of landscape complexity
    scheduler = GeneticScheduler(number_of_generation=1,
                                 mutate_order=0.05,
                                 mutate_resources=0.005,
                                 size_of_population=1)

    # Get information about created LandscapeConfiguration
    platform_number = len(landscape.platforms)
    is_all_nodes_have_materials = all([node.work_unit.need_materials() for node in wg.nodes])
    print(f'LandscapeConfiguration: {platform_number} platforms, '
          f'All nodes have materials: {is_all_nodes_have_materials}')

    # Get list with the Contractor object, which can satisfy the created WorkGraph's resources requirements
    contractors = [get_contractor_by_wg(wg)]

    project = DefaultInputPipeline() \
        .wg(wg) \
        .contractors(contractors) \
        .landscape(landscape) \
        .schedule(scheduler) \
        .visualization('2023-01-01')[0] \
        .show_gant_chart()