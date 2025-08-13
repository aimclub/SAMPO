from operator import itemgetter
from typing import Optional

import numpy as np
from toposort import CircularDependencyError

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.utilities.priority import extract_priority_groups_from_nodes
from sampo.schemas.graph import GraphNode
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator


class TopologicalScheduler(GenericScheduler):
    """
    Scheduler, that represent 'WorkGraph' in topological order.
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.Topological,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=AverageReqResourceOptimizer(),
                         timeline_type=MomentumTimeline,
                         prioritization_f=self._topological_sort,
                         optimize_resources_f=self.get_default_res_opt_function(lambda _: Time(0)),
                         work_estimator=work_estimator)

    def _shuffle(self, nodes: set[str]) -> list[str]:
        """
        Shuffle nodes that are on the same level.

        :param nodes: list of nodes
        :return: list of shuffled indices
        """
        return list(nodes)

    # noinspection PyMethodMayBeStatic
    def _topological_sort(self,
                          head_nodes: list[GraphNode],
                          node_id2parent_ids: dict[str, set[str]],
                          node_id2child_ids: dict[str, set[str]],
                          work_estimator: WorkTimeEstimator) -> list[GraphNode]:

        ordered_nodes = []

        priority_groups = extract_priority_groups_from_nodes(head_nodes)

        id2node = {node.id: node for node in head_nodes}

        for _, priority_group in sorted(priority_groups.items(), key=itemgetter(0)):
            priority_group_set = set(node.id for node in priority_group)
            priority_group_dict = {k.id: node_id2parent_ids[k.id].intersection(priority_group_set)
                                   for k in priority_group}
            tsorted_node_ids: list[str] = [node_id
                                           for level in toposort(priority_group_dict)
                                           for node_id in self._shuffle(level)]

            ordered_nodes.extend([id2node[node] for node in tsorted_node_ids])

        # validate_order(ordered_nodes)

        return ordered_nodes


def toposort(data):
    """\
    Dependencies are expressed as a dictionary whose keys are items
    and whose values are a set of dependent items. Output is a list of
    sets in topological order. The first set consists of items with no
    dependences, each subsequent set consists of items that depend upon
    items in the preceeding sets.
    """

    # Special case empty input.
    if len(data) == 0:
        return

    # Copy the input so as to leave it unmodified.
    # Discard self-dependencies and copy two levels deep.
    data = {item: set(e for e in dep if e != item) for item, dep in data.items()}

    # Find all items that don't depend on anything.
    extra_items_in_deps = {value for values in data.values() for value in values} - set(
        data.keys()
    )
    # The line below does N unions of value sets, which is much slower than the
    # set comprehension above which does 1 union of N value sets. The speedup
    # gain is around 200x on a graph with 190k nodes.
    # extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())

    # Add empty dependences where needed.
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {
            item: (dep - ordered) for item, dep in data.items() if item not in ordered
        }
    if len(data) != 0:
        raise CircularDependencyError(data)


def validate_order(order: list[GraphNode]):
    seen = set()

    for node in order:
        assert all(parent in seen for parent in node.parents)
        seen.add(node)


class RandomizedTopologicalScheduler(TopologicalScheduler):
    """
    Scheduler, that represent 'WorkGraph' in topological order with random.
    """

    def __init__(self,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 random_seed: Optional[int] = None):
        super().__init__(work_estimator=work_estimator)
        self._random_state = np.random.RandomState(random_seed)

    def _shuffle(self, nodes: set[str]) -> list[str]:
        """
        Shuffle nodes that are on the same level.

        :param nodes: list of nodes
        :return: list of shuffled indices
        """
        nds = list(nodes)
        indices = np.arange(len(nds))
        self._random_state.shuffle(indices)
        return [nds[ind] for ind in indices]
