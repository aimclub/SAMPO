from typing import Callable
from uuid import uuid4

from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit


def calculate_working_time_cascade(node: GraphNode, appointed_worker: list[Worker],
                                   work_estimator: WorkTimeEstimator = None) -> Time:
    """
    Calculate the working time of the appointed workers at a current job for prioritization.
    O(1) - at worst case |inseparable_edges|

    :param node: the target node
    :return: working time
    """
    if node.is_inseparable_son():
        # time of employment of resources and task execution is calculated only for the first job
        # in the chain of connected inextricably
        return Time(0)
    common_time = node.work_unit.estimate_static(appointed_worker, work_estimator)  # working time

    # calculation of the time for all work_units inextricably linked to the given
    while node.is_inseparable_parent():
        node = node.inseparable_son
        common_time += node.work_unit.estimate_static(appointed_worker, work_estimator)
    return common_time


def calculate_working_time(work_unit: WorkUnit, appointed_worker: list[Worker],
                           work_estimator: WorkTimeEstimator = None) -> Time:
    """
    Calculate the working time of the appointed workers at a current job for final schedule

    :return: working time
    """
    return work_unit.estimate_static(appointed_worker, work_estimator)  # working time


PRIORITY_DELTA = 1


def work_priority(node: GraphNode,
                  comp_cost: Callable[[GraphNode, list[Worker], WorkTimeEstimator], Time],
                  work_estimator: WorkTimeEstimator = None) -> float:
    """
    Calculate the average time to complete the work when assigning the minimum and maximum number of employees
    for the correct calculations of rank in prioritization
    O(sum_of_max_counts_of_workers) of current work

    :param node: the target node
    :param work_estimator:
    :param comp_cost: function for calculating working time (calculate_working_time)
    :return: average working time
    """

    work_unit = node.work_unit

    passed_workers_min = [Worker(str(uuid4()), req.kind, req.min_count)
                          for req in work_unit.worker_reqs]

    passed_workers_max = [Worker(str(uuid4()), req.kind, req.max_count)
                          for req in work_unit.worker_reqs]

    res = (comp_cost(node, passed_workers_min, work_estimator) +
           comp_cost(node, passed_workers_max, work_estimator)) / 2

    return res + PRIORITY_DELTA
