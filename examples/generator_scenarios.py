import random
from itertools import chain

from sampo.generator import SimpleSynthetic, get_contractor_by_wg
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.graph import WorkGraph
from sampo.structurator import graph_in_graph_insertion, work_graph_ids_simplification, graph_restructuring
from sampo.utilities.visualization.work_graph import work_graph_fig

rand = random.Random(10)
p_rand = SimpleSynthetic(rand=231)
wg = p_rand.work_graph(top_border=3000)
contractors = [p_rand.contactor(i) for i in range(10, 31, 10)]
schedule = HEFTScheduler().schedule(wg, contractors)

print(len(wg.nodes))
print("\nDefault contractors")
print(f"Execution time: {schedule.execution_time}")

print("\nContractor by work graph")
for pack_counts in []:#[1, 2, 10]:
    contractors = [get_contractor_by_wg(wg, scaler=pack_counts)]
    execution_time = HEFTScheduler().schedule(wg, contractors).execution_time
    print(f"Execution time: {execution_time}, pack count: {pack_counts}")

print("\nNames extension")
names_wg = len(set(n.work_unit.name for n in wg.nodes))
new_wg = extend_names(500, wg, rand)
names_new_wg = len(set(n.work_unit.name for n in new_wg.nodes))
print(f"works in origin: {names_wg}, pack count: {names_new_wg}")


print("\nResource extension")

res_names_wg = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
new_wg = extend_resources(100, wg, rand)
res_names_new_wg = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in new_wg.nodes])))
print(f"resources in origin: {res_names_wg}, in new: {res_names_new_wg}")

print("\nCheck gen contractor by wg extension")

res_names_new_c = len(set(get_contractor_by_wg(new_wg, scaler=1).workers.keys()))
print(f"works in new by wg: {res_names_new_wg}, by contractor: {res_names_new_c}")


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