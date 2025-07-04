from uuid import uuid4

import numpy as np

from sampo.schemas import GraphNode, Worker, WorkTimeEstimator
from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade


def get_chain_duration(node: GraphNode, assigned_workers_amounts: np.ndarray, work_estimator: WorkTimeEstimator) -> int:
    work_unit = node.work_unit

    passed_workers = [Worker(str(uuid4()), req.kind, assigned_amount)
                      for req, assigned_amount in zip(work_unit.worker_reqs, assigned_workers_amounts)]

    chain_duration = calculate_working_time_cascade(node, passed_workers, work_estimator).value + 1

    return chain_duration
