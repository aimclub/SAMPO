from typing import List

from schemas.contractor import Contractor
from schemas.graph import WorkGraph


def check_all_workers_have_same_qualification(wg: WorkGraph, contractors: List[Contractor]):
    # 1. all workers of the same category belonging to the same contractor should have the same characteristics
    for c in contractors:
        assert all(ws.count >= 1 for _, ws in c.workers.items()), \
            f"There should be only one worker for the same worker category"

    # добавляем агентов в словарь
    agents = {}
    for contractor in contractors:
        for name, val in contractor.workers.items():
            if name[0] not in agents:
                agents[name[0]] = 0
            agents[name[0]] += val.count
    # 2. all tasks should have worker reqs that can be satisfied by at least one contractor
    for v in wg.nodes:
        assert any(
            all(c.worker_types[wreq.type][0].count
                >= (wreq.min_count + min(agents[wreq.type], wreq.max_count)) // 2
                for wreq in v.work_unit.worker_reqs)
            for c in contractors
        ), f"The work unit with id {v.work_unit.id} cannot be satisfied by any contractors"
