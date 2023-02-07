from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple

from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator


class Timeline(ABC):

    @abstractmethod
    def schedule(self,
                 task_index: int,
                 node: GraphNode,
                 node2swork: Dict[GraphNode, ScheduledWork],
                 passed_agents: List[Worker],
                 contractor: Contractor,
                 assigned_start_time: Optional[Time],
                 assigned_time: Optional[Time],
                 work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        ...

    def find_min_start_time(self,
                            node: GraphNode,
                            worker_team: List[Worker],
                            node2swork: Dict[GraphNode, ScheduledWork],
                            parent_time: Time = Time(0),
                            work_estimator: Optional[WorkTimeEstimator] = None) -> Time:
        """
        Computes start time, max parent time, contractor and exec times for given node

        :param worker_team: list of passed workers. Should be IN THE SAME ORDER AS THE CORRESPONDING WREQS
        :param node:
        :param node2swork:
        :param parent_time:
        :param work_estimator:
        :return:
        """
        return self.find_min_start_time_with_additional(node, worker_team, node2swork, None,
                                                        parent_time, work_estimator)[0]

    @abstractmethod
    def find_min_start_time_with_additional(self,
                                            node: GraphNode,
                                            worker_team: List[Worker],
                                            node2swork: Dict[GraphNode, ScheduledWork],
                                            assigned_start_time: Optional[Time] = None,
                                            assigned_parent_time: Time = Time(0),
                                            work_estimator: Optional[WorkTimeEstimator] = None) \
            -> Tuple[Time, Time, Dict[GraphNode, Tuple[Time, Time]]]:
        ...

    @abstractmethod
    def update_timeline(self,
                        task_index: int,
                        finish_time: Time,
                        node: GraphNode,
                        node2swork: Dict[GraphNode, ScheduledWork],
                        worker_team: List[Worker]):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...
