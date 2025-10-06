from collections import defaultdict
from typing import Iterable

from toposort import toposort_flatten

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
    Identifies 'head nodes' in a WorkGraph and reconstructs their inter-node dependencies.

    Head nodes are defined as the first nodes of inseparable chains or standalone nodes
    that are not part of an inseparable chain (i.e., they are not 'inseparable sons').
    This function effectively flattens the graph by treating inseparable chains as
    single logical entities represented by their head node, and then re-establishes
    parent-child relationships between these head nodes.

    Args:
        wg: The `WorkGraph` to analyze.

    Returns:
        A tuple containing:
            - A list of `GraphNode` objects representing the head nodes,
              sorted in topological order based on their reconstructed dependencies.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'parent' head nodes. These represent external dependencies
              where a parent of any node within the current head node's inseparable
              chain belongs to another head node's chain.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'child' head nodes. Similar to parents, these represent
              external dependencies where a child of any node within the current
              head node's inseparable chain belongs to another head node's chain.
    """
    return get_head_nodes_with_connections_mappings_nodes(wg.nodes)


def get_head_nodes_with_connections_mappings_nodes(nodes: list[GraphNode]) -> tuple[list[GraphNode], dict[str, set[str]], dict[str, set[str]]]:
    """
    Identifies 'head nodes' in a WorkGraph and reconstructs their inter-node dependencies.

    Head nodes are defined as the first nodes of inseparable chains or standalone nodes
    that are not part of an inseparable chain (i.e., they are not 'inseparable sons').
    This function effectively flattens the graph by treating inseparable chains as
    single logical entities represented by their head node, and then re-establishes
    parent-child relationships between these head nodes.

    Args:
        nodes: The `WorkGraph` to analyze.

    Returns:
        A tuple containing:
            - A list of `GraphNode` objects representing the head nodes,
              sorted in topological order based on their reconstructed dependencies.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'parent' head nodes. These represent external dependencies
              where a parent of any node within the current head node's inseparable
              chain belongs to another head node's chain.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'child' head nodes. Similar to parents, these represent
              external dependencies where a child of any node within the current
              head node's inseparable chain belongs to another head node's chain.
    """
    # Filter the work graph nodes to identify all 'head nodes'.
    # A head node is one that is not an 'inseparable son', meaning it's either
    # the start of an inseparable chain or a standalone node.
    nodes = [node for node in nodes if not node.is_inseparable_son()]
    node_dict = {node.id: node for node in nodes}

    # Construct a mapping from any node within an inseparable chain to its
    # corresponding head node.
    node2inseparable_parent = {child: node for node in nodes for child in node.get_inseparable_chain_with_self()}

    # Reconstruct parent-child relationships between the identified head nodes.
    # For each head node, gather all direct parents of any node within its
    # inseparable chain. Then, map these direct parents back to their
    # respective head nodes using `node2inseparable_parent`.
    node_id2parent_ids = {node.id: set(
        node2inseparable_parent[parent].id  # Get the head node ID for the actual parent
        for inseparable in node.get_inseparable_chain_with_self()  # Iterate through all parts of head node's chain
        for parent in inseparable.parents  # Get direct parents of each part
    ) - {node.id}  # Remove self-references
                          for node in nodes}

    # Reconstruct child-parent relationships in the same manner as parent-child.
    # For each head node, gather all direct children of any node within its
    # inseparable chain. Then, map these direct children back to their
    # respective head nodes using `node2inseparable_parent`.
    node_id2child_ids = {node.id: set(
        node2inseparable_parent[child].id  # Get the head node ID for the actual child
        for inseparable in node.get_inseparable_chain_with_self()  # Iterate through all parts of head node's chain
        for child in inseparable.children_set  # Get direct children of each part
    ) - {node.id}  # Remove self-references
                         for node in nodes}

    # Perform a topological sort on the head nodes based on their new parent dependencies.
    # This ensures the returned list of head nodes is ordered correctly for scheduling or analysis.
    tsorted_nodes_ids = toposort_flatten(node_id2parent_ids, sort=True)

    # Map the sorted head node IDs back to their corresponding GraphNode objects.
    tsorted_nodes = [node_dict[node_id] for node_id in tsorted_nodes_ids]

    return tsorted_nodes, node_id2parent_ids, node_id2child_ids
