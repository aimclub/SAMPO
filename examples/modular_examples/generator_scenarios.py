import random
from itertools import chain

from sampo.generator import SimpleSynthetic, get_contractor_by_wg
from sampo.generator.pipeline.extension import extend_names, extend_resources
from sampo.scheduler.heft.base import HEFTScheduler

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