class BinsPheromones:
    
    def __init__(self, 
                 n_tasks: int, 
                 n_bins: int, 
                 random, 
                 start_amount: float,
                 evaporation_rate: float):
        # number of choices to make
        self.n_tasks = n_tasks
        # number of bins for each choice
        self.n_bins = n_bins

        # start from uniform (for now)
        self.bins_pheromones = start_amount * np.ones((n_tasks, n_bins))

        # ph = np.zeros((n_tasks, n_bins))
        # for i, (a, b) in enumerate(zip(np.linspace(0, 1, n_bins), np.linspace(1, 0, n_bins))):
        #     ph[:, i] = np.linspace(a, b, n_tasks)
        # self.bins_pheromones = start_amount * ph


        # evaporation rate controls speed of changing old pheromones to new
        self.evaporation_rate = evaporation_rate
        # to make sure the seed is right everywhere, random is an attribute
        self.random = random

    def weighted_choice(self, array: list, weights: list):
        probabilities = weights / np.sum(weights)
        choice = self.random.choice(array, p=probabilities)
        return choice

    def generate_one_path(self) -> list[int]:
        """Create one random path using current pheromones"""

        priority_list = np.zeros(self.n_tasks)
        for i in range(0, self.n_tasks):
            
            # if self.random.random() < 0.01:
            #     weights = np.ones(self.n_bins) 
            # else:
            #     weights = self.bins_pheromones[i]

            chosen_bin = self.weighted_choice(
                np.arange(self.n_bins),
                self.bins_pheromones[i]
            )

            value = int(chosen_bin)

            priority_list[i] = value

        # priority_list += self.random.random(self.n_tasks)

        return priority_list

    def update_pheromones(self, priorities_array, fitness_array, reference_array) -> None:
        """Evaporate current pheromones"""
        
        is_good_solution = fitness_array < np.quantile(reference_array, 0.10)
        if sum(is_good_solution) == 0:
            return None
        priorities_best = priorities_array[is_good_solution]


        new_pheromone = (0 * self.bins_pheromones)
        for path in priorities_best:
            for node, value in enumerate(path):
                new_pheromone[node, int(value)] += 100 


        # fig = px.imshow(
        #     new_pheromone.T,
        #     color_continuous_scale=["white", "blue", "navy"],
        #     template="plotly_dark"
        # )
        # fig.update_layout(title_text="Diff")
        # fig.show()

        self.bins_pheromones = (
            (1 - self.evaporation_rate) * self.bins_pheromones + 
            self.evaporation_rate * new_pheromone
        )

        # fig = px.imshow(
        #     self.bins_pheromones.T,
        #     color_continuous_scale=["white", "blue", "navy"],
        #     template="plotly_dark"
        # )
        # fig.update_layout(title_text="Value")
        # fig.show()



    def stretch(self, k):
        self.n_bins *= k
        self.bins_pheromones = np.repeat(self.bins_pheromones, repeats=k, axis=1) / k

    def renormalize(self):
        for i in range(self.bins_pheromones.shape[0]):
            self.bins_pheromones[i] = self.bins_pheromones[i] / self.bins_pheromones[i].max() * 10



    # @staticmethod
    # def quantile_based_reward(fitness_array: list[float], 
    #                           reference_array: list[float], 
    #                           k_groups: int = 10) -> list[float]:
    #     # get quantiles of fitness of last few generations
    #     bins = np.quantile(reference_array, np.linspace(1, 0, k_groups+1))
    #     # small fix for borders
    #     bins[0], bins[-1] = np.inf, -np.inf
    #     quantiles = np.digitize(fitness_array, bins=bins)
    #     # reward = rank of the group, squared
    #     # seems ok for now, balance between better rewards for top
    #     # but also not too harsh for others
    #     pheromone_amount = (quantiles >= 10) * 25

    #     # # big reward for the best from reference (last few generations)
    #     # best_in_reference = max(reference_array)
    #     # for i, fitness in enumerate(fitness_array):
    #     #     if fitness >= best_in_reference:
    #     #         pheromone_amount[i] *= 2
        
    #     return pheromone_amount


