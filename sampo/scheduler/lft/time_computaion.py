from uuid import uuid4

import numpy as np

from sampo.schemas import GraphNode, Worker, WorkTimeEstimator


def work_duration(node: GraphNode, assigned_workers_amounts: np.ndarray, work_estimator: WorkTimeEstimator) -> list[int]:
    work_unit = node.work_unit

    passed_workers = [Worker(str(uuid4()), req.kind, assigned_amount)
                      for req, assigned_amount in zip(work_unit.worker_reqs, assigned_workers_amounts)]

    chain_durations = [work_estimator.estimate_time(dep_node.work_unit, passed_workers).value
                       for dep_node in node.get_inseparable_chain_with_self()]

    return chain_durations
