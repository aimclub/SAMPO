from sampo.generator import SimpleSynthetic, get_contractor_by_wg
from sampo.scheduler.heft.base import HEFTScheduler

p_rand = SimpleSynthetic(seed=231)
wg = p_rand.work_graph(top_border=400)
contractors = [p_rand.contactor(i) for i in range(10, 31, 10)]
schedule = HEFTScheduler().schedule(wg, contractors)

print(len(wg.nodes))
print(schedule.execution_time)
print("\nDefault contractors")
print(f"Execution time: {schedule.execution_time}")

print("\nContractor by work graph")
for pack_counts in [1, 2, 10]:
    contractors = [get_contractor_by_wg(wg, scaler=pack_counts)]
    execution_time = HEFTScheduler().schedule(wg, contractors).execution_time
    print(f"Execution time: {execution_time}, pack count: {pack_counts}")
