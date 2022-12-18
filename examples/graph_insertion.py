from matplotlib import pyplot as plt

from sampo.generator import SimpleSynthetic
from sampo.schemas.graph import WorkGraph
from sampo.structurator import graph_in_graph_insertion, work_graph_ids_simplification, graph_restructuring
from sampo.utilities.visualization.work_graph import work_graph_fig


def plot_wg(wg: WorkGraph) -> None:
    _ = work_graph_fig(wg, (14, 8), legend_shift=4, show_names=True, text_size=4)
    plt.show()


srand = SimpleSynthetic(34)
wg_master = srand.work_graph(cluster_counts=1)
wg_slave = srand.work_graph(cluster_counts=1)

union_wg = graph_in_graph_insertion(wg_master, wg_master.start, wg_master.finish, wg_slave)
union_wg_simply = work_graph_ids_simplification(union_wg, id_offset=1000)
union_wg_restructured = graph_restructuring(union_wg_simply)

plot_wg(wg_master)
plot_wg(wg_slave)
plot_wg(union_wg)
plot_wg(union_wg_simply)
plot_wg(union_wg_restructured)