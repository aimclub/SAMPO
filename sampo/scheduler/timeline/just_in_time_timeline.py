from typing import Optional

from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.material_timeline import SupplyTimeline
from sampo.scheduler.timeline.zone_timeline import ZoneTimeline
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


class JustInTimeTimeline(Timeline):
    """
    Timeline that stored the time of resources release.
    For each contractor and worker type store a descending list of pairs of time and
    number of available workers of this type of this contractor.
    """

    def __init__(self, worker_pool: WorkerContractorPool, landscape: LandscapeConfiguration):
        self._timeline = {}
        # stacks of time(Time) and count[int]
        for worker_type, worker_offers in worker_pool.items():
            for worker_offer in worker_offers.values():
                self._timeline[worker_offer.get_agent_id()] = [(Time(0), worker_offer.count)]

        self._material_timeline = SupplyTimeline(landscape)
        self.zone_timeline = ZoneTimeline(landscape.zone_config)

    def find_min_start_time_with_additional(self, node: GraphNode,
                                            worker_team: list[Worker],
                                            node2swork: dict[GraphNode, ScheduledWork],
                                            spec: WorkSpec,
                                            assigned_start_time: Time | None = None,
                                            assigned_parent_time: Time = Time(0),
                                            work_estimator: WorkTimeEstimator = DefaultWorkEstimator()) \
            -> tuple[Time, Time, dict[GraphNode, tuple[Time, Time]]]:
        """
        Define the nearest possible start time for the current job. It is equal the max value from:
        1. end time of all parent tasks,
        2. time previous job off all needed workers to complete the current task.

        :param assigned_parent_time: minimum start time
        :param assigned_start_time:
        :param node: the GraphNode whose minimum time we are trying to find
        :param worker_team: the worker team under testing
        :param node2swork: dictionary, that match GraphNode to ScheduleWork respectively
        :param spec: given work specification
        :param work_estimator: function that calculates execution time of the GraphNode
        :return: start time, end time, None(exec_times not needed in this timeline)
        """
        # if current job is the first
        if not node2swork:
            max_material_time = self._material_timeline.find_min_material_time(node.id, assigned_parent_time,
                                                                               node.work_unit.need_materials(),
                                                                               node.work_unit.workground_size)

            max_zone_time = self.zone_timeline.find_min_start_time(node.work_unit.zone_reqs, max_material_time, Time(0))

            return max_zone_time, max_zone_time, None
        # define the max end time of all parent tasks
        max_parent_time = max(node.min_start_time(node2swork), assigned_parent_time)
        # define the max agents time when all needed workers are off from previous tasks
        max_agent_time = Time(0)

        if spec.is_independent:
            # grab from the end
            for worker in worker_team:
                offer_stack = self._timeline[worker.get_agent_id()]
                max_agent_time = max(max_agent_time, offer_stack[0][0])
        else:
            # grab from whole sequence
            # for each resource type
            for worker in worker_team:
                needed_count = worker.count
                offer_stack = self._timeline[worker.get_agent_id()]
                # traverse list while not enough resources and grab it
                ind = len(offer_stack) - 1
                while needed_count > 0:
                    offer_time, offer_count = offer_stack[ind]
                    max_agent_time = max(max_agent_time, offer_time)

                    if needed_count < offer_count:
                        offer_count = needed_count
                    needed_count -= offer_count
                    ind -= 1

        c_st = max(max_agent_time, max_parent_time)

        new_finish_time = c_st
        for dep_node in node.get_inseparable_chain_with_self():
            # set start time as finish time of original work
            # set finish time as finish time + working time of current node with identical resources
            # (the same as in original work)
            # set the same workers on it
            # TODO Decide where this should be
            dep_parent_time = dep_node.min_start_time(node2swork)

            dep_st = max(new_finish_time, dep_parent_time)
            working_time = work_estimator.estimate_time(dep_node.work_unit, worker_team)
            new_finish_time = dep_st + working_time

        exec_time = new_finish_time - c_st

        max_material_time = self._material_timeline.find_min_material_time(node.id, c_st,
                                                                           node.work_unit.need_materials(),
                                                                           node.work_unit.workground_size)

        max_zone_time = self.zone_timeline.find_min_start_time(node.work_unit.zone_reqs, c_st, exec_time)

        c_st = max(c_st, max_material_time, max_zone_time)

        c_ft = c_st + exec_time
        return c_st, c_ft, None

    def can_schedule_at_the_moment(self,
                                   node: GraphNode,
                                   worker_team: list[Worker],
                                   spec: WorkSpec,
                                   node2swork: dict[GraphNode, ScheduledWork],
                                   start_time: Time,
                                   exec_time: Time) -> bool:
        if spec.is_independent:
            # squash all the timeline to the last point
            for worker in worker_team:
                worker_timeline = self._timeline[(worker.contractor_id, worker.name)]
                last_cpkt_time, _ = worker_timeline[0]
                if last_cpkt_time >= start_time:
                    return False
            return True
        else:
            # checking edges
            for dep_node in node.get_inseparable_chain_with_self():
                for p in dep_node.parents:
                    if p != dep_node.inseparable_parent:
                        swork = node2swork.get(p, None)
                        if swork is None or swork.finish_time >= start_time:
                            return False

            max_agent_time = Time(0)
            for worker in worker_team:
                needed_count = worker.count
                offer_stack = self._timeline[worker.get_agent_id()]
                # traverse list while not enough resources and grab it
                ind = len(offer_stack) - 1
                while needed_count > 0:
                    offer_time, offer_count = offer_stack[ind]
                    max_agent_time = max(max_agent_time, offer_time)

                    if needed_count < offer_count:
                        offer_count = needed_count
                    needed_count -= offer_count
                    ind -= 1

            if not max_agent_time <= start_time:
                return False

            if not self._material_timeline.can_schedule_at_the_moment(node.id, start_time,
                                                                      node.work_unit.need_materials(),
                                                                      node.work_unit.workground_size):
                return False
            if not self.zone_timeline.can_schedule_at_the_moment(node.work_unit.zone_reqs, start_time, exec_time):
                return False

            return True

    def update_timeline(self,
                        finish_time: Time,
                        exec_time: Time,
                        node: GraphNode,
                        worker_team: list[Worker],
                        spec: WorkSpec):
        """
        Adds given `worker_team` to the timeline at the moment `finish`

        :param finish_time:
        :param exec_time:
        :param node:
        :param worker_team:
        :param spec: work specification
        :return:
        """
        if spec.is_independent:
            # squash all the timeline to the last point
            for worker in worker_team:
                worker_timeline = self._timeline[(worker.contractor_id, worker.name)]
                count_workers = sum([count for _, count in worker_timeline])
                worker_timeline.clear()
                worker_timeline.append((finish_time + 1, count_workers))
        else:
            # For each worker type consume the nearest available needed worker amount
            # and re-add it to the time when current work should be finished.
            # Addition performed as step in bubble-sort algorithm.
            for worker in worker_team:
                needed_count = worker.count
                worker_timeline = self._timeline[(worker.contractor_id, worker.name)]
                # Consume needed workers
                while needed_count > 0:
                    next_time, next_count = worker_timeline.pop()
                    if next_count > needed_count or len(worker_timeline) == 0:
                        worker_timeline.append((next_time, next_count - needed_count))
                        break
                    needed_count -= next_count

                # Add to the right place
                # worker_timeline.append((finish + 1, worker.count))
                # worker_timeline.sort(reverse=True)
                worker_timeline.append((finish_time + 1, worker.count))
                ind = len(worker_timeline) - 1
                while ind > 0 and worker_timeline[ind][0] > worker_timeline[ind - 1][0]:
                    worker_timeline[ind], worker_timeline[ind - 1] = worker_timeline[ind - 1], worker_timeline[ind]
                    ind -= 1

    def schedule(self,
                 node: GraphNode,
                 node2swork: dict[GraphNode, ScheduledWork],
                 workers: list[Worker],
                 contractor: Contractor,
                 spec: WorkSpec,
                 assigned_start_time: Optional[Time] = None,
                 assigned_time: Optional[Time] = None,
                 assigned_parent_time: Time = Time(0),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        inseparable_chain = node.get_inseparable_chain_with_self()

        start_time = assigned_start_time if assigned_start_time is not None \
            else self.find_min_start_time(node, workers,
                                          node2swork,
                                          spec,
                                          assigned_parent_time,
                                          work_estimator)

        if assigned_time is not None:
            exec_times = {n: (Time(0), assigned_time // len(inseparable_chain))
                          for n in inseparable_chain}
            return self._schedule_with_inseparables(node, node2swork, workers, contractor, spec,
                                                    inseparable_chain, start_time, exec_times, work_estimator)
        else:
            return self._schedule_with_inseparables(node, node2swork, workers, contractor, spec,
                                                    inseparable_chain, start_time, {}, work_estimator)

    def _schedule_with_inseparables(self,
                                    node: GraphNode,
                                    node2swork: dict[GraphNode, ScheduledWork],
                                    workers: list[Worker],
                                    contractor: Contractor,
                                    spec: WorkSpec,
                                    inseparable_chain: list[GraphNode],
                                    start_time: Time,
                                    exec_times: dict[GraphNode, tuple[Time, Time]],
                                    work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        """
        Makes ScheduledWork object from `GraphNode` and worker list, assigned `start_end_time`
        and adds it to given `node2swork`. Also does the same for all inseparable nodes starts from this one

        :param node:
        :param node2swork:
        :param workers:
        :param contractor:
        :param spec:
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
            max_parent_time = dep_node.min_start_time(node2swork)

            if dep_node.is_inseparable_son():
                assert max_parent_time >= node2swork[dep_node.inseparable_parent].finish_time

            if dep_node in exec_times:
                lag, working_time = exec_times[dep_node]
            else:
                lag, working_time = 0, work_estimator.estimate_time(node.work_unit, workers)
            c_st = max(c_ft + lag, max_parent_time)
            new_finish_time = c_st + working_time

            deliveries, _, new_finish_time = self._material_timeline.deliver_materials(dep_node.id, c_st,
                                                                                       new_finish_time,
                                                                                       dep_node.work_unit.need_materials(),
                                                                                       dep_node.work_unit.workground_size)

            node2swork[dep_node] = ScheduledWork(work_unit=dep_node.work_unit,
                                                 start_end_time=(c_st, new_finish_time),
                                                 workers=workers,
                                                 contractor=contractor,
                                                 materials=deliveries)
            # change finish time for using workers
            c_ft = new_finish_time

        zones = [zone_req.to_zone() for zone_req in node.work_unit.zone_reqs]
        self.update_timeline(c_ft, c_ft - start_time, node, workers, spec)
        node2swork[node].zones_pre = self.zone_timeline.update_timeline(len(node2swork), zones, start_time,
                                                                        c_ft - start_time)
        return c_ft
