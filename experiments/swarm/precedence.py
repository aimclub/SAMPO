import numpy as np

class PrecedenceManager:
    """Responsible for managing precedence relations"""
    
    def __init__(self,
                 successors: dict[int, set], 
                 predecessors: dict[int, int]
                 ):
        # number of nodes in a graph
        # not really necessary, but keep it just in case
        self.n_nodes = len(successors)
        # dictionary {node id: set of successors/children}
        self.successors = successors
        # dictionary {node id: number of predecessors}
        # we don't need exact predecessors, just the number is enough
        self.n_predecessors = {key: len(value) for key, value in predecessors.items()}

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

    def get_activity_list(self, priorities_array: list[float]):
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
        
        return np.array(activity_list)

    def convert_al_to_valid_order(self, activity_list: list[int]):
        # first, reset any previos info
        self.reset_completion()
        # order of jobs added
        valid_activity_list = []
        # now, constructing the list
        for _ in range(self.n_nodes):
            what_can_start = self.what_can_start
            for i in activity_list:
                if i in what_can_start:
                    next_to_add = i
                    break
            
            # set selected job as completed
            self.set_complete(next_to_add)
            valid_activity_list.append(next_to_add)
        
        return np.array(valid_activity_list)

    # converting to valid order is the main reason for this class
    def __call__(self, priorities_array: list[float]):
        return self.convert_priorities_to_valid_order(priorities_array)





        