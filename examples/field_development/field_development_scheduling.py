from operator import itemgetter

from matplotlib import pyplot as plt
from sampo.structurator.graph_insertion import graph_in_graph_insertion
from sampo.structurator.light_modifications import work_graph_ids_simplification

from sampo.generator import get_contractor_by_wg  # Warning!!! sampo~=0.1.1.77
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.graph import WorkGraph
from sampo.structurator.base import graph_restructuring
from sampo.utilities.visualization.work_graph import work_graph_fig

path = "../"

wg1 = WorkGraph.load(path, f"cluster_25_block_1_wg")
wg2 = WorkGraph.load(path, f"cluster_25_block_2_wg")

# Insert wg2 into wg1
union_wg = graph_in_graph_insertion(wg1, wg1.start, wg1.finish, wg2)

# Changing the id to numeric instead of uuid (id is still a string)
union_wg = work_graph_ids_simplification(union_wg, id_offset=1000)

# We transform the edges in the graph, it is necessary!
# after the transformation, some work is divided into several so that only the FS edges remain
structured_wg = graph_restructuring(union_wg, use_lag_edge_optimization=True)
_ = work_graph_fig(structured_wg, (20, 10), legend_shift=4, show_names=True, text_size=6)
plt.show()
contractors = [get_contractor_by_wg(union_wg, scaler=3)]
schedule = HEFTScheduler().schedule(structured_wg, contractors)
# the schedule stores the divided works after restructuring, merge them using this function and work further with df
schedule_df = schedule.merged_stages_datetime_df("2022-12-21")

# add all vertices to the database except wg.start and wg.finish

print(schedule_df)

# add a number to the second level of the hierarchy, that is,
# "Обустройство кустовых площадок:Монтаж силового кабеля 1", "Обустройство кустовых площадок:Монтаж силового кабеля 2"
hierarchical_names = {
    "cluster_25_block_2": "Обустройство кустовых площадок:Монтаж силового кабеля",
    "cluster_25_block_1": "Обустройство кустовых площадок:Монтаж металлоконструкций",
    "cluster_25": "Обустройство кустовых площадок:Кустовая площадка",
    "power_line": "Прокладка электролиний:Электролиния",
}