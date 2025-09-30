from sampo.schemas.graph import GraphNode
from sampo.schemas.time import Time


def get_exec_times_from_assigned_time_for_chain(inseparable_chain: list[GraphNode],
                                                assigned_time: Time) -> dict[GraphNode, Time]:
    """
    Distributes a given total execution time among work nodes in an inseparable chain.

    The time distribution is proportional to each node's volume, ensuring that
    the entire `assigned_time` is utilized. Any rounding discrepancies are
    allocated to the last node in the chain.

    Args:
        inseparable_chain: A list of nodes representing an inseparable sequence of work units.
        assigned_time: The total `Time` allocated for the entire chain's execution.

    Returns:
        A dictionary mapping each `GraphNode` to a tuple `(lag, node_execution_time)`.
        `lag` is always `Time(0)` as the chain is inseparable, and
        `node_execution_time` is the calculated execution time for that specific node.
    """
    total_volume = sum(n.work_unit.volume for n in inseparable_chain)

    # Handle the edge case where total_volume is zero or negative.
    # In this scenario, time is distributed equally among nodes,
    # with any remainder allocated to the last node.
    if total_volume <= 0:
        node_time = assigned_time // len(inseparable_chain)
        # Distribute time equally among all but the last node
        exec_times = {n: node_time for n in inseparable_chain[:-1]}
        # Assign remaining time to the last node
        exec_times[inseparable_chain[-1]] = assigned_time - node_time * len(exec_times)
        return exec_times

    exec_times: dict[GraphNode, Time] = {}
    remaining_time = assigned_time  # Initialize remaining time to distribute

    # Iterate through all nodes except the last one.
    # Time is distributed based on the node's volume relative to the *remaining* total volume.
    for i, n in enumerate(inseparable_chain[:-1]):
        # Calculate the proportion of the current node's volume to the remaining total volume.
        volume_proportion = 0
        if total_volume > 0:
            volume_proportion = n.work_unit.volume / total_volume

        # Calculate execution time for the current node and convert to integer.
        # This takes a portion of the *remaining* time.
        exec_time = int(remaining_time.value * volume_proportion)
        exec_times[n] = Time(exec_time)

        # Deduct the current node's volume and allocated time from the totals
        total_volume -= n.work_unit.volume
        remaining_time -= exec_time

    # The last node receives all remaining time to account for any rounding errors
    # during the distribution to previous nodes.
    exec_times[inseparable_chain[-1]] = remaining_time
    return exec_times
