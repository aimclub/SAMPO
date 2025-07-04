from collections import defaultdict
from typing import Iterable

from sampo.schemas import Worker, Contractor, WorkGraph, GraphNode
from sampo.schemas.types import WorkerName, ContractorName

WorkerContractorPool = dict[WorkerName, dict[ContractorName, Worker]]


def get_worker_contractor_pool(contractors: Iterable[Contractor]) -> WorkerContractorPool:
    """
    Gets worker-contractor dictionary from contractors list.
    Alias for frequently used functionality.

    :param contractors: list of all the considered contractors
    :return: dictionary of workers by worker name, next by contractor id
    """
    worker_pool = defaultdict(dict)
    for contractor in contractors:
        for name, worker in contractor.workers.items():
            worker_pool[name][contractor.id] = worker.copy()
    return worker_pool


def get_head_nodes_with_connections_mappings(wg: WorkGraph
                                             ) -> tuple[list[GraphNode], dict[str, set[str]], dict[str, set[str]]]:
    """
    Returns data structures for working with head nodes (nodes that do not contain inseparable children).

    Args:
        wg: A work graph.

    Returns:
        A tuple containing:
            - A list containing only the head nodes
              (first node of inseparable chain or node that is not a part of inseparable chain), i.e.
              a list of nodes from work graph without inseparable children.
            - A dictionary that maps head node IDs to the set of head node IDs to which the parent edge now belongs.
            - A dictionary that maps head node IDs to the set of head node IDs to which the child edge now belongs.
    """
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    # construct inseparable_child -> inseparable_parent mapping
    node2inseparable_parent = {child: node for node in nodes for child in node.get_inseparable_chain_with_self()}

    # here we aggregate information about relationships from the whole inseparable chain
    node_id2parent_ids = {node.id: set(node2inseparable_parent[parent].id
                                       for inseparable in node.get_inseparable_chain_with_self()
                                       for parent in inseparable.parents_set) - {node.id}
                          for node in nodes}

    node_id2child_ids = {node.id: set(node2inseparable_parent[child].id
                                      for inseparable in node.get_inseparable_chain_with_self()
                                      for child in inseparable.children_set) - {node.id}
                         for node in nodes}

    return nodes, node_id2parent_ids, node_id2child_ids
