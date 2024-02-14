import warnings

from matplotlib import pyplot as plt

from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.structurator.base import graph_restructuring
from sampo.utilities.visualization.base import VisualizationMode
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig
from sampo.utilities.visualization.work_graph import work_graph_fig

warnings.filterwarnings("ignore")  # for matplotlib warning suppression

# Set this flag if you want to use manually prepared contractors' info
# (otherwise, contractors will be generated automatically to satisfy the uploaded WorkGraph's requirements)
use_contractors_from_file = False

# Uploading WorkGraph's data from prepared JSON file
field_development_wg = WorkGraph.load("./", "field_development_tasks_structure")

if use_contractors_from_file:
    # Uploading Contractor's data from prepared JSON file
    contractors = Contractor.load("./", "field_development_contractors_info")
else:
    # Automated contractors' generation by the uploaded WorkGraph
    contractors = [get_contractor_by_wg(field_development_wg, scaler=3)]

# WorkGraph structure optimization
structured_wg = graph_restructuring(field_development_wg, use_lag_edge_optimization=True)

# WorkGraph structure visualization
_ = work_graph_fig(structured_wg, (20, 10), legend_shift=4, show_names=True, text_size=6)
plt.show()

# Set up attributes for the algorithm and results presentation
scheduler_type = HEFTScheduler()  # Set up scheduling algorithm
graph_structure_optimization = True  # Graph structure optimization flag
csv_required = False  # Saving schedule to .csv file flag
json_required = True  # Saving schedule to .json file flag

visualization_mode = VisualizationMode.ShowFig  # Set up visualization type
gant_chart_filename = './output/schedule_gant_chart.png'  # Set up gant chart file name (if necessary)

start_date = "2023-01-01"  # Set up the project's start date

# Schedule field development tasks
schedule = scheduler_type.schedule(structured_wg, contractors, validate=True)[0]
schedule_df = schedule.merged_stages_datetime_df(start_date)

# Schedule's gant chart visualization
gant_fig = schedule_gant_chart_fig(schedule_df,
                                   fig_file_name=gant_chart_filename,
                                   visualization=visualization_mode,
                                   remove_service_tasks=True)

# # Visualize schedule
# schedule_gant_chart_fig(schedule_df, VisualizationMode.ShowFig)
