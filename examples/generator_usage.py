from sampo.generator import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler

p_rand = SimpleSynthetic(seed=231)
wg = p_rand.work_graph(top_border=400)
contractors = [p_rand.contactor(i) for i in range(10, 31, 10)]
schedule = HEFTScheduler().schedule(wg, contractors)

print(len(wg.nodes))
print(wg)
print(schedule.to_schedule_work_dict)