import numpy as np

class NestedBinsPheromones:
    
    def __init__(self, 
                 n_tasks: int, 
                 n_bins: int, 
                 n_sub_bins: int, 
                 random, 
                 start_amount: float = 0, 
                 evaporation_rate: float = 0.25):
        # number of choices to make
        self.n_tasks = n_tasks
        # number of bins and subbins for each choice
        self.n_bins = n_bins
        self.n_sub_bins = n_sub_bins

        # start from uniform (for now) 
        self.bins_pheromones = \
            start_amount * np.ones((n_tasks, n_bins))
        self.sub_bins_pheromones = \
            start_amount * np.ones((n_tasks, n_bins, n_sub_bins))

        # evaporation rate controls speed of changing old pheromones to new
        self.evaporation_rate = evaporation_rate
        # to make sure the seed is right everywhere, random is an attribute
        self.random = random

    def weighted_choice(self, array: list, weights: list):
        # probabilities = weights / np.sum(weights)
        probabilities = self.softmax(weights)
        choice = self.random.choice(array, p=probabilities)
        return choice

    def generate_one_path(self) -> list[tuple[int, int]]:
        """Create one random path using current pheromones"""

        priority_list = []
        for i in range(0, self.n_tasks):
            # main priority value
            chosen_bin = self.weighted_choice(
                np.arange(self.n_bins), 
                self.bins_pheromones[i]
            )
            chosen_bin = int(chosen_bin)
            # sub priority value
            chosen_sub_bin = self.weighted_choice(
                np.arange(self.n_sub_bins), 
                self.sub_bins_pheromones[i][chosen_bin]
            )
            chosen_sub_bin = int(chosen_sub_bin)
            
            priority_list.append((chosen_bin, chosen_sub_bin))

        return np.array(priority_list)

    def get_pheromones(self, 
                       path_list: list, 
                       pheromone_amount_list: list[float]
                       ) -> None:
        """
        Add new pheromones based on one generation
        path_list: list of paths 
        pheromone_amount_list: list of amounts to add
        """
        new_pheromone = (0 * self.bins_pheromones)
        new_sub_pheromone = (0 * self.sub_bins_pheromones)

        for path, pheromone_amount in zip(path_list, pheromone_amount_list):

            for node, value in enumerate(path):
                new_pheromone[node, value[0]] += pheromone_amount                
                new_sub_pheromone[node, value[0], value[1]] += pheromone_amount

        return new_pheromone, new_sub_pheromone

    def update_pheromones(self, path_list, pheromone_amount_list) -> None:
        """Evaporate current pheromones"""
        new_pheromone, new_sub_pheromone = self.get_pheromones(path_list, pheromone_amount_list)
        
        # fig = px.imshow(new_pheromone.T)
        # fig.update_layout(title_text="Diff")
        # fig.show()

        self.bins_pheromones = (
            (1 - self.evaporation_rate) * self.bins_pheromones + 
            self.evaporation_rate * new_pheromone
        )
        self.sub_bins_pheromones = (
            (1 - self.evaporation_rate) * self.sub_bins_pheromones +
            self.evaporation_rate * new_sub_pheromone
        )

        # fig = px.imshow(self.bins_pheromones.T)
        # fig.update_layout(title_text="Current")
        # fig.show()


    @staticmethod
    def softmax(array, beta=1):
        w = np.exp(beta * array)
        return w / np.sum(w)



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

        priority_list = []
        for i in range(0, self.n_tasks):
            
            # smoothed = np.correlate(
            #     self.bins_pheromones[i], 
            #     (0.25, 0.50, 0.25), 
            #     mode="same"
            # )

            chosen_bin = self.weighted_choice(
                np.arange(self.n_bins),
                self.bins_pheromones[i]
                # (self.bins_pheromones[i] ** 0.90) * (smoothed ** 0.10)
            )
            
            priority_list.append(int(chosen_bin))

        return np.array(priority_list)

    def update_pheromones(self, priorities_array, fitness_array, reference_array) -> None:
        """Evaporate current pheromones"""
        
        is_good_solution = fitness_array < np.quantile(reference_array, 0.10)
        if sum(is_good_solution) == 0:
            return None
        priorities_best = priorities_array[is_good_solution]
        
        # priorities_std = np.std(priorities_best, axis=0)
        # is_std_low = priorities_std < 5  # np.quantile(priorities_std, 0.50)
        # if sum(is_std_low) == 0:
        #     return None


        # fig = px.scatter(
        #     x=np.std(self.bins_pheromones, axis=1),
        #     y=priorities_std,
        #     template="plotly_white",
        #     trendline="ols"
        # )
        # fig.update_layout(height=400, width=400)
        # fig.show()


        new_pheromone = (0 * self.bins_pheromones)
        for path in priorities_best:
            for node, value in enumerate(path):
                # if is_std_low[node]:
                new_pheromone[node, value] += 25   


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
        #     self.bins_pheromones.T / self.bins_pheromones.sum(axis=1),
        #     color_continuous_scale=["white", "blue", "navy"],
        #     template="plotly_dark"
        # )
        # fig.update_layout(title_text="Value")
        # fig.show()


    def stretch(self, k):
        self.n_bins *= k
        self.bins_pheromones = np.repeat(self.bins_pheromones, repeats=k, axis=1)


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



