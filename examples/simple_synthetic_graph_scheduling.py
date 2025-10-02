from itertools import chain

from sampo.generator.environment.contractor_by_wg import get_contractor_by_wg
from sampo.utilities.visualization.base import VisualizationMode

from sampo.utilities.visualization.schedule import schedule_gant_chart_fig

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.time import Time

# Set up attributes for the generated synthetic graph
synth_works_top_border = 2000
synth_unique_works = 300
synth_resources = 100

# Set up scheduling algorithm and project's start date
scheduler = HEFTScheduler()
start_date = "2023-01-01"

# Set up visualization mode (ShowFig or SaveFig) and the gant chart file's name (if SaveFig mode is chosen)
visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = './output/synth_schedule_gant_chart.png'

# Generate synthetic graph with the given approximate works count,
# number of unique works names and number of unique resources
srand = SimpleSynthetic(rand=31)
wg = srand.advanced_work_graph(works_count_top_border=synth_works_top_border,
                               uniq_works=synth_unique_works,
                               uniq_resources=synth_resources)

# Get information about created WorkGraph's attributes
works_count = len(wg.nodes)
work_names_count = len(set(n.work_unit.model_name for n in wg.nodes))
res_kind_count = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
print(works_count, work_names_count, res_kind_count)

# Check the validity of the WorkGraph's attributes
assert (works_count <= synth_works_top_border * 1.1)
assert (work_names_count <= synth_works_top_border)
assert (res_kind_count <= synth_works_top_border)

# Get list with the Contractor object, which can satisfy the created WorkGraph's resources requirements
contractors = [get_contractor_by_wg(wg)]

# Schedule works
schedule = scheduler.schedule(wg, contractors)[0]
schedule_df = schedule.merged_stages_datetime_df(start_date)
# Schedule's gant chart visualization
gant_fig = schedule_gant_chart_fig(schedule_df,
                                   fig_file_name=gant_chart_filename,
                                   visualization=visualization_mode,
                                   remove_service_tasks=True)
print(schedule.execution_time)

# Check the validity of the schedule's time
assert schedule.execution_time != Time.inf(), f'Scheduling failed on {scheduler.scheduler_type.name}'

