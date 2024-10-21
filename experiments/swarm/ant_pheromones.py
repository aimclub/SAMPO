import numpy as np

class AntPheromones:
    
    def __init__(self, n_tasks, n_bins, n_subbins, start_amount=500):
        self.pheromone_matrix = \
            start_amount * np.ones((n_tasks, n_bins))
        self.subpheromone_matrix = \
            start_amount * np.ones((n_tasks, n_bins, n_subbins))

        # # for later
        # elif pheromone_init == "start_linear":
        #     starts = np.linspace(0, 1, n_bins)
        #     ends = np.linspace(1, 0, n_bins)
        #     pheromone_matrix = np.zeros((n_tasks, n_bins))
        #     for i, (start, end) in enumerate(zip(starts, ends)):
        #         pheromone_matrix[:, i] = np.linspace(start, end, n_tasks)

        #     self.pheromone_matrix = 500 * pheromone_matrix
        #     self.subpheromone_matrix = \
        #         np.random.uniform(500, 500, (n_tasks, n_bins, n_subbins))
    
    def get_pheromones(self):
        return (self.pheromone_matrix, self.subpheromone_matrix)

    def get_zeros_pheromones(self):
        # basically just getting the shape of pheromone matrix
        return (0 * self.pheromone_matrix, 0 * self.subpheromone_matrix)
    
    def add_pheromones(self, 
                       new_pheromones=None, 
                       new_subpheromones=None, 
                       evaporation=0.70):
        
        if new_pheromones is not None:
            self.pheromone_matrix = (
                evaporation * self.pheromone_matrix + 
                (1-evaporation) * new_pheromones
            )
        
        if new_subpheromones is not None:
            self.subpheromone_matrix = (
                evaporation * self.subpheromone_matrix + 
                (1-evaporation) * new_subpheromones
            )

    def give_everyone_pheromones(self, amount):
        self.pheromone_matrix += amount
        self.subpheromone_matrix += amount


# class SlidingPheromones:
    
#     def __init__(self, n_bins, n_subbins):
#         self.pheromone_matrix_list = [
#             np.random.uniform(10, 10, (N_TASKS_WITHDUMMY, n_bins))
#             for i in range(20)
#         ]
#         self.subpheromone_matrix_list = [
#             np.random.uniform(10, 10, (N_TASKS_WITHDUMMY, n_bins, n_subbins))
#             for i in range(20)
#         ]
    
#     @staticmethod
#     def get_weights(n=20, alpha=1.00):
#         weights = np.ones(n) * alpha ** np.arange(n)
#         return weights[::-1]
    
#     def get_pheromones(self):
#         weights = self.get_weights(n=20)
#         last_few_pheromone_matrix = self.pheromone_matrix_list[-20:]
#         last_few_subpheromone_matrix = self.subpheromone_matrix_list[-20:]
#         return (
#             sum( a*b for a, b in zip(last_few_pheromone_matrix, weights) ),
#             sum( a*b for a, b in zip(last_few_subpheromone_matrix, weights) )
#         )
    
#     def add_pheromones(self, new_pheromones, new_subpheromones):
#         self.pheromone_matrix_list.append(new_pheromones)
#         self.subpheromone_matrix_list.append(new_subpheromones)







