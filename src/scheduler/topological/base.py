from typing import List, Dict, Set, Optional, Any, Iterable

import numpy as np
from toposort import toposort_flatten, toposort

from external.estimate_time import WorkTimeEstimator
from scheduler.base import Scheduler
from scheduler.base import SchedulerType
from scheduler.utils.momentum_timeline import schedule, prepare_worker, create_timeline
from scheduler.utils.multi_contractor import get_best_contractor_and_worker_borders
from schemas.contractor import Contractor, get_worker_contractor_pool
from schemas.graph import GraphNode, WorkGraph
from schemas.requirements import WorkerReq
from schemas.resources import Worker
from schemas.schedule import Schedule
from schemas.scheduled_work import ScheduledWork
from utilities.validation import validate_schedule


class TopologicalScheduler(Scheduler):

    def __init__(self, scheduler_type: SchedulerType = SchedulerType.Topological,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type=scheduler_type,
                         work_estimator=work_estimator)

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 validate: bool = False) \
            -> Schedule:
        # Checking pre-conditions for this scheduler_topological to be applied
        # check_all_workers_have_same_qualification(wg, contractors)

        tsorted_nodes: List[GraphNode] = self._topological_sort(wg)

        schedule = Schedule.from_scheduled_works(
            self.build_scheduler(tsorted_nodes, contractors, self.work_estimator), wg
        )

        # check the validity received scheduler
        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule

    # noinspection PyMethodMayBeStatic
    def _topological_sort(self, wg: WorkGraph) -> List[GraphNode]:
        node2ind = {node: i for i, node in enumerate(wg.nodes)}

        dependents: Dict[int, Set[int]] = {
            node2ind[v]: {node2ind[node] for node in v.parents}
            for v in wg.nodes
        }

        tsorted_nodes_indices: List[int] = toposort_flatten(dependents, sort=True)
        tsorted_nodes = [wg.nodes[i] for i in tsorted_nodes_indices]
        return tsorted_nodes

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, tasks: List[GraphNode], contractors: List[Contractor],
                        work_estimator: WorkTimeEstimator = None) \
            -> Iterable[ScheduledWork]:
        """
        Builds a schedule from a list of tasks where all dependents tasks are guaranteed to be later
        in the sequence than their dependencies
        :param work_estimator:
        :param tasks: list of tasks ordered by some algorithm according to their dependencies and priorities
        :param wg: graph of tasks to be executed
        :param contractors: pools of workers available for execution
        :return: a schedule
        """
        # now we may work only with workers that have
        # only workers with the same productivity
        # (e.g. for each specialization each contractor has only one worker object)
        # check_all_workers_have_same_qualification(wg, contractors)

        # data structure to hold scheduled tasks
        node2swork: Dict[GraphNode, ScheduledWork] = dict()

        timeline = create_timeline(tasks, contractors)

        # we can get agents here, because they are always same and not updated
        agents = get_worker_contractor_pool(contractors)

        # We allocate resources for the whole inseparable chain, when we process the first node in it.
        # So, we will store IDs of non-head nodes in such chains to skip them.
        # Note that tasks are already topologically ordered,
        # i.e., the first node in a chain is always processed before its children

        skipped_inseparable_children: Set[str] = set()
        # scheduling all the tasks in a one-by-one manner
        for i, node in enumerate(tasks):
            # skip, if this node was processed as a part of an inseparable chin previously
            if node.id in skipped_inseparable_children:
                continue

            # 0. find, if the node starts an inseparable chain

            inseparable_chain = node.get_inseparable_chain()
            if inseparable_chain:
                skipped_inseparable_children.update((ch.id for ch in inseparable_chain))
            whole_work_nodes = inseparable_chain if inseparable_chain else [node]

            _, _, contractor, _ = \
                get_best_contractor_and_worker_borders(agents, contractors, node.work_unit.worker_reqs)

            passed_agents = [prepare_worker(agents, req, contractor.id, count_getter=get_worker_count)
                             for i, req in enumerate(node.work_unit.worker_reqs)]

            schedule(i, node, node2swork, whole_work_nodes, timeline, passed_agents, contractor, work_estimator)

        return node2swork.values()


def get_worker_count(req: WorkerReq, worker_of_contractor: Worker):
    return (req.min_count + min(worker_of_contractor.count, req.max_count)) // 2
    # return min_req


class RandomizedTopologicalScheduler(TopologicalScheduler):
    def __init__(self, work_estimator: Optional[WorkTimeEstimator or None] = None,
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator)
        self._random_state = np.random.RandomState(random_seed)

    def _topological_sort(self, wg: WorkGraph) -> List[GraphNode]:
        def shuffle(nodes: Set[GraphNode]) -> List[GraphNode]:
            nds = list(nodes)
            indices = np.arange(len(nds))
            self._random_state.shuffle(indices)
            return [nds[ind] for ind in indices]

        dependents: Dict[GraphNode, Set[GraphNode]] = {v: v.parents_set for v in wg.nodes}
        tsorted_nodes: List[GraphNode] = [
            node for level in toposort(dependents)
            for node in shuffle(level)
        ]

        # print([node.work_unit.id for node in tsorted_nodes])
        return tsorted_nodes
