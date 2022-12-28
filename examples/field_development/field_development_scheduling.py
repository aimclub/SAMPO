import json
from operator import itemgetter

from matplotlib import pyplot as plt

from examples.field_development.utils import data_to_work_graph
from sampo.structurator.graph_insertion import graph_in_graph_insertion
from sampo.structurator.light_modifications import work_graph_ids_simplification

from sampo.generator import get_contractor_by_wg  # Warning!!! sampo~=0.1.1.77
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.graph import WorkGraph
from sampo.structurator.base import graph_restructuring
from sampo.utilities.visualization.work_graph import work_graph_fig


object_construction_wg_path = "field_object_construction_graph_structure.json"
with open(object_construction_wg_path, encoding="utf8") as json_data:
    graph_json_obj = json.load(json_data)
# object_construction_wg_data = WrapperMap(graph_json_obj['work_graph'])
object_construction_wg = data_to_work_graph(graph_json_obj['work_graph'])

contractors = [get_contractor_by_wg(object_construction_wg, scaler=3)]

structured_wg = graph_restructuring(object_construction_wg, use_lag_edge_optimization=True)
_ = work_graph_fig(structured_wg, (20, 10), legend_shift=4, show_names=True, text_size=6)
plt.show()

schedule = HEFTScheduler().schedule(structured_wg, contractors)
schedule_df = schedule.merged_stages_datetime_df("2022-12-21")
print(schedule_df)
