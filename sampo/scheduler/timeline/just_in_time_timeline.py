from typing import Dict, List, Tuple, Optional

from sampo.scheduler.heft.time_computaion import calculate_working_time
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.types import AgentId


class JustInTimeTimeline(Timeline):

    def __init__(self, worker_pool: WorkerContractorPool):
        self._timeline = {}
        # stacks of time(Time) and count[int]
        for worker_type, worker_offers in worker_pool.items():
            for worker_offer in worker_offers.values():
                self._timeline[worker_offer.get_agent_id()] = [(Time(0), worker_offer.count)]

    def find_min_start_time(self, node: GraphNode, worker_team: List[Worker],
                            id2swork: Dict[GraphNode, ScheduledWork],
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        """
        Define the nearest possible start time for current job. It is equal the max value from:
        1. end time of all parent tasks
        2. time previous job off all needed workers to complete current task

        :param node: target node
        :param worker_team: worker team under testing
        :param id2swork:
        :param work_estimator:
        :return: found time, queue of employees without assigned workers
        """
        # if current job is the first
        if len(id2swork) == 0:
            return Time(0)
        # define the max end time of all parent tasks
        max_parent_time = max([id2swork[parent_node].finish_time
                               for parent_node in node.parents], default=Time(0))
        # define the max agents time when all needed workers are off from previous tasks
        max_agent_time = Time(0)

        # For each resource type
        for worker in worker_team:
            needed_count = worker.count
            offer_stack = self._timeline[worker.get_agent_id()]
            # Traverse list while not enough resources and grab it
            ind = len(offer_stack) - 1
            while needed_count > 0:
                offer_time, offer_count = offer_stack[ind]
                max_agent_time = max(max_agent_time, offer_time)

                if needed_count < offer_count:
                    offer_count = needed_count
                needed_count -= offer_count
                ind -= 1

        return max(max_agent_time, max_parent_time)

    def update_timeline(self,
                        task_index: int,
                        finish_time: Time,
                        node: GraphNode,
                        node2swork: Dict[GraphNode, ScheduledWork],
                        worker_team: List[Worker]):
        """
        Adds given `worker_team` to the timeline at the moment `finish`
        :param task_index:
        :param finish_time:
        :param node:
        :param node2swork:
        :param worker_team:
        :return:
        """
        # For each worker type consume the nearest available needed worker amount
        # and re-add it to the time when current work should be finished.
        # Addition performed as step in bubble-sort algorithm.
        for worker in worker_team:
            needed_count = worker.count
            worker_timeline = self._timeline[(worker.contractor_id, worker.name)]
            # Consume needed workers
            while needed_count > 0:
                next_time, next_count = worker_timeline.pop()
                if next_count > needed_count:
                    worker_timeline.append((next_time, next_count - needed_count))
                    break
                needed_count -= next_count

            # Add to the right place
            # worker_timeline.append((finish, worker.count))
            # worker_timeline.sort(reverse=True)
            worker_timeline.append((finish_time, worker.count))
            ind = len(worker_timeline) - 1
            while ind > 0 and worker_timeline[ind][0] > worker_timeline[ind - 1][0]:
                worker_timeline[ind], worker_timeline[ind - 1] = worker_timeline[ind - 1], worker_timeline[ind]
                ind -= 1

    def schedule(self,
                 task_index: int,
                 node: GraphNode,
                 id2swork: Dict[GraphNode, ScheduledWork],
                 workers: List[Worker],
                 contractor: Contractor,
                 assigned_start_time: Optional[Time] = None,
                 assigned_time: Optional[Time] = None,
                 work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        inseparable_chain = node.get_inseparable_chain_with_self()
        st = assigned_start_time if assigned_start_time else self.find_min_start_time(node, workers, id2swork)
        if assigned_time:
            exec_times = {n: (Time(0), assigned_time // len(inseparable_chain))
                          for n in inseparable_chain}
            return self._schedule_with_inseparables(id2swork, workers, contractor, inseparable_chain, st, exec_times,
                                                    work_estimator)
        else:
            return self._schedule_with_inseparables(id2swork, workers, contractor, inseparable_chain, st, {},
                                                    work_estimator)

    def __getitem__(self, item: AgentId):
        return self._timeline[item]

    @staticmethod
    def _schedule_with_inseparables(id2swork: Dict[GraphNode, ScheduledWork],
                                    workers: List[Worker],
                                    contractor: Contractor,
                                    inseparable_chain: List[GraphNode],
                                    start_time: Time,
                                    exec_times: Dict[GraphNode, Tuple[Time, Time]],
                                    work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        """
        Makes ScheduledWork object from `GraphNode` and worker list, assigned `start_end_time`
        and adds it ti given `id2swork`. Also does the same for all inseparable nodes starts from this one
        :param id2swork:
        :param workers:
        :param contractor:
        :param inseparable_chain:
        :param start_time:
        :param exec_times:
        :param work_estimator:
        :return:
        """
        c_ft = start_time
        for dep_node in inseparable_chain:
            # set start time as finish time of original work
            # set finish time as finish time + working time of current node with identical resources
            # (the same as in original work)
            # set the same workers on it
            # TODO Decide where this should be
            max_parent_time = max((id2swork[pnode].finish_time
                                   for pnode in dep_node.parents),
                                  default=Time(0))
            working_time = exec_times.get(dep_node, None)
            start_time = max(c_ft, max_parent_time)
            if working_time is None:
                working_time = calculate_working_time(dep_node.work_unit, workers, work_estimator)
            new_finish_time = start_time + working_time

            id2swork[dep_node] = ScheduledWork(work_unit=dep_node.work_unit,
                                               start_end_time=(start_time, new_finish_time),
                                               workers=workers,
                                               contractor=contractor)
            # change finish time for using workers
            c_ft = new_finish_time

        return c_ft
