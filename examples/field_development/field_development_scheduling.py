from matplotlib import pyplot as plt
from sampo.utilities.visualization.base import VisualizationMode

from sampo.generator import get_contractor_by_wg  # Warning!!! sampo~=0.1.1.77
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.graph import WorkGraph
from sampo.structurator.base import graph_restructuring
from sampo.utilities.visualization.work_graph import work_graph_fig
from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

import warnings

warnings.filterwarnings("ignore")  # for matplotlib warning suppression

# Uploading WorkGraph data from prepared JSON file
field_development_wg = WorkGraph.load("./", "field_development_tasks_structure")

# Use this code for automated contractors' generation by the uploaded WorkGraph
contractors = [get_contractor_by_wg(field_development_wg, scaler=3)]

# WorkGraph structure optimization
structured_wg = graph_restructuring(field_development_wg, use_lag_edge_optimization=True)

# WorkGraph structure visualization
_ = work_graph_fig(structured_wg, (20, 10), legend_shift=4, show_names=True, text_size=6)
plt.show()

# Set up the project's start date
start_date = "2022-12-21"

# Schedule field development tasks
schedule = HEFTScheduler().schedule(structured_wg, contractors)
schedule_df = schedule.merged_stages_datetime_df(start_date)

# Visualize schedule
schedule_gant_chart_fig(schedule_df, VisualizationMode.ShowFig)
