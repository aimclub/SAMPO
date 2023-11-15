from abc import ABC, abstractmethod
from typing import Optional

from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


class Timeline(ABC):
    """
    Entity that saves info on the use of resources over time.
    Timeline provides opportunities to work with GraphNodes and resources over time.
    """

    @abstractmethod
    def schedule(self,
                 node: GraphNode,
                 node2swork: dict[GraphNode, ScheduledWork],
                 passed_agents: list[Worker],
                 contractor: Contractor,
                 spec: WorkSpec,
                 assigned_start_time: Time | None = None,
                 assigned_time: Time | None = None,
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> Time:
        """
        Schedules the given `GraphNode` using passed agents, spec and times.
        If start time not passed, it should be computed as minimum work start time.
        :return: scheduled finish time of given work
        """
        ...

    def find_min_start_time(self,
                            node: GraphNode,
                            worker_team: list[Worker],
                            node2swork: dict[GraphNode, ScheduledWork],
                            spec: WorkSpec,
                            parent_time: Time = Time(0),
                            work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) -> Time:
        """
        Computes start time, max parent time, contractor and exec times for given node.

        :param worker_team: list of passed workers. It Should be IN THE SAME ORDER AS THE CORRESPONDING WREQS
        :param node: the GraphNode whose minimum time we are trying to find
        :param node2swork: dictionary, that match GraphNode to ScheduleWork respectively
        :param spec: specification for given `GraphNode`
        :param parent_time: the minimum start time
        :param work_estimator: function that calculates execution time of the GraphNode
        :return: minimum time
        """
        return self.find_min_start_time_with_additional(node, worker_team, node2swork, spec, None,
                                                        parent_time, work_estimator)[0]

    @abstractmethod
    def find_min_start_time_with_additional(self,
                                            node: GraphNode,
                                            worker_team: list[Worker],
                                            node2swork: dict[GraphNode, ScheduledWork],
                                            spec: WorkSpec,
                                            assigned_start_time: Optional[Time] = None,
                                            assigned_parent_time: Time = Time(0),
                                            work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
            -> tuple[Time, Time, dict[GraphNode, tuple[Time, Time]]]:
        ...

    @abstractmethod
    def can_schedule_at_the_moment(self,
                                   node: GraphNode,
                                   worker_team: list[Worker],
                                   spec: WorkSpec,
                                   node2swork: dict[GraphNode, ScheduledWork],
                                   start_time: Time,
                                   exec_time: Time) -> bool:
        """
        Returns the ability of scheduling given `node` at the `start_time` moment
        """
        ...

    @abstractmethod
    def update_timeline(self,
                        finish_time: Time,
                        exec_time: Time,
                        node: GraphNode,
                        worker_team: list[Worker],
                        spec: WorkSpec):
        ...
