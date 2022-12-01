from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator


class Timeline(ABC):

    def schedule(self,
                 task_index: int,
                 node: GraphNode,
                 node2swork: Dict[GraphNode, ScheduledWork],
                 passed_agents: List[Worker],
                 contractor: Contractor,
                 assigned_time: Optional[Time],
                 work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        inseparable_chain = node.get_inseparable_chain()
        inseparable_chain = inseparable_chain if inseparable_chain else [node]

        return self.schedule_impl(task_index, node, node2swork, passed_agents, contractor,
                                  inseparable_chain, assigned_time, work_estimator)

    @abstractmethod
    def schedule_impl(self,
                      task_index: int,
                      node: GraphNode,
                      node2swork: Dict[GraphNode, ScheduledWork],
                      passed_agents: List[Worker],
                      contractor: Contractor,
                      inseparable_chain: List[GraphNode],
                      assigned_time: Optional[Time],
                      work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        ...

    @abstractmethod
    def find_min_start_time(self, node: GraphNode, worker_team: List[Worker],
                            id2swork: Dict[str, ScheduledWork],
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        ...

    @abstractmethod
    def update_timeline(self, finish: Time, worker_team: List[Worker]):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...
