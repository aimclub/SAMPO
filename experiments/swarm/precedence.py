
class PrecedenceManager:
    """Responsible for managing precedence relations"""
    
    def __init__(self, 
                 n_nodes: int, 
                 successors: dict[int, set], 
                 n_predecessors: dict[int, int]
                 ):
        # number of nodes in a graph
        # not really necessary, but keep it just in case
        self.n_nodes = n_nodes
        # dictionary {node id: set of successors/children}
        self.successors = successors
        # dictionary {node id: number of predecessors}
        # we don't need exact predecessors, just the number is enough
        self.n_predecessors = n_predecessors

        # track how many predecessors were completed
        # will be set by the reset method
        self.n_completed_predecessors = None
        # set of jobs that have completed predecessors and were not finished already
        # will be set by the reset method
        self.what_can_start = None 
        # reset the states of nodes
        self.reset_completion()

    def reset_completion(self):
        """Clear the info about what jobs were completed"""
        # why use a class with reset(), and not just a function?
        # 1) saving info about successors and predecessors in one place
        # 2) a bit more readable thanks to splitting logic to smaller methods

        # set number of completed predecessors to zero for everyone
        self.n_completed_predecessors = {node_id: 0 for node_id in self.n_predecessors.keys()}
        # at the start, only jobs with no predecessors can start
        self.what_can_start = {node_id for node_id, k in self.n_predecessors.items() if (not k)}

    @classmethod
    def from_workgraph(cls, wg):
        """Create a class from sampo workgraph"""
        n_nodes = len(wg.nodes)
        # use index of a node instead of node itself
        # later we need index of a job as int (because matrices need index)
        # so might as well convert it to int right away
        node2index = {node: i for i, node in enumerate(wg.nodes)}
        successors = {node2index[node]: {node2index[i] for i in node.children_set} for node in wg.nodes}
        n_predecessors = {node2index[node]: len(node.parents_set) for node in wg.nodes}
        return cls(n_nodes, successors, n_predecessors)

    def set_complete(self, node_index: int):
        """Update state after completing a job"""
        # if job was finished, it cannot be started again
        self.what_can_start.remove(node_index)
        # for each successor of the finished node
        for successor in self.successors[node_index]:
            # now we have one more completed predecessor
            self.n_completed_predecessors[successor] += 1
            # now check if that it was the last predecessors we needed
            if self.n_completed_predecessors[successor] == self.n_predecessors[successor]:
                # if it is, then the job can start
                self.what_can_start.add(successor)


    def convert_priorities_to_valid_order(self, priorities_array: list[int] | list[float] | list[tuple]):
        """
        Convert list of priorities for jobs to a valid order
        If priorities_array[A] > priorities_array[B], then
            job A will be added first to order
        (if precedence constraints will allow it)
        """
        # first, reset any previos info
        self.reset_completion()
        # order of jobs added
        activity_list = []
        # now, constructing the list
        for _ in range(self.n_nodes):
            # we add job that is:
            next_to_add = max(
                # 1) has all predecessors finished
                self.what_can_start,
                # 2) has the maximum priority among others
                key=lambda x: priorities_array[x]
            )
            # set selected job as completed
            self.set_complete(next_to_add)
            activity_list.append(next_to_add)
        
        return activity_list

    # converting to valid order is the main reason for this class
    def __call__(self, priorities_array: list[int] | list[float] | list[tuple]):
        return self.convert_priorities_to_valid_order(priorities_array)


    def get_what_can_start(self):
        return sorted(list(self.what_can_start))



# # Saved for later, might be useful for integration
# def get_predecessors_dict_from_adj_matrix(adj_matrix):
#     return {
#         node: [
#             i for i, bit in enumerate(adj_matrix[:, node])
#             if bit and (i != node)
#         ]
#         for node in range(len(adj_matrix))
#     }




