from uuid import uuid4

from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.time_estimator import WorkTimeEstimator

PRIORITY_DELTA = 1


def work_min_max_duration(node: GraphNode, work_estimator: WorkTimeEstimator) -> tuple[int, int]:
    work_unit = node.work_unit

    passed_workers_min = [Worker(str(uuid4()), req.kind, req.min_count)
                          for req in work_unit.worker_reqs]

    passed_workers_max = [Worker(str(uuid4()), req.kind, req.max_count)
                          for req in work_unit.worker_reqs]

    min_duration = work_estimator.estimate_time(node.work_unit, passed_workers_max)
    max_duration = work_estimator.estimate_time(node.work_unit, passed_workers_min)

    return min_duration + PRIORITY_DELTA, max_duration + PRIORITY_DELTA
