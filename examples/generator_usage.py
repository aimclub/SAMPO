from sampo.generator.pipeline.project import get_graph
from sampo.generator.enviroment.contractor import get_contractor
from sampo.scheduler.heft.base import HEFTScheduler

wg = get_graph(top_border=400)
contractors = [get_contractor(i) for i in range(10, 31, 10)]
schedule = HEFTScheduler().schedule(wg, contractors)

print(len(wg.nodes))
print(wg)
print(schedule.to_schedule_work_dict)