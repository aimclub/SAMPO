
class PrecedenceManager:
    
    def __init__(self, predecessors: dict):
        self.predecessors = predecessors
        self.is_completed = {
            i: False 
            for i in predecessors.keys()
        }
        self.order_added = []
        # self.candidates_were = []
        
    def can_start(self, job_id):
        predecessors = self.predecessors[job_id]
        return all(
            self.is_completed[i]
            for i in predecessors
        )
    
    def what_can_start(self):
        candidates = [
            i for i in self.is_completed.keys()
            if self.can_start(i) and not self.is_completed[i]
        ]
        # self.candidates_were.append(candidates)
        return candidates
    
    def set_complete(self, job_id):
        self.is_completed[job_id] = True
        self.order_added.append(job_id)
        
    def are_all_completed(self):
        return all(self.is_completed.values())


class RandomKeyFitness:

    def __init__(self, launcher, predecessors_dict):
        self.launcher = launcher
        self.predecessors_dict = predecessors_dict

    def __call__(self, random_key_array):
        jobs_sorted = self.sort_random_key(random_key_array, self.predecessors_dict)
        makespan = self.launcher.get_makespan(jobs_sorted)
        return makespan

    @staticmethod
    def sort_random_key(random_key_array, predecessors_dict):
        
        if len(random_key_array) != len(predecessors_dict):
            raise Exception("len() of arguments must be equal")
        
        pm = PrecedenceManager(predecessors_dict)
        for _ in range(len(random_key_array)):
            what_can_start = pm.what_can_start()
            chosen_job = max(
                what_can_start,
                key=lambda x: random_key_array[x]
            )
            pm.set_complete(chosen_job)
    
        if not pm.are_all_completed():
            raise Exception("Not all jobs were marked as completed")
    
        return pm.order_added


# def get_predecessors_dict_from_adj_matrix(adj_matrix):
#     return {
#         job_id: [
#             i for i, bit in enumerate(adj_matrix[:, job_id])
#             if bit and (i != job_id)
#         ]
#         for job_id in range(len(adj_matrix))
#     }










