from itertools import chain

from sampo.generator import SimpleSynthetic, get_contractor_by_wg
from sampo.scheduler.heft.base import HEFTScheduler

srand = SimpleSynthetic(34)
wg = srand.advanced_work_graph(works_count_top_border=2000, uniq_works=300, uniq_resources=100)

works_count = len(wg.nodes)
work_names_count = len(set(n.work_unit.name for n in wg.nodes))
res_kind_count = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
print(works_count, work_names_count, res_kind_count)
# 2036 248 100

contractors = [get_contractor_by_wg(wg, i) for i in [5, 2, 10]]
schedule = HEFTScheduler().schedule(wg, contractors)
print(schedule.execution_time)
# 11843
