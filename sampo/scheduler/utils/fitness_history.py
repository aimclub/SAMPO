import numpy as np   # for calculating aggregate functions
import pandas as pd  # for saving data to csv

class FitnessHistory:
    """
    Class to track fitness and other stats for genetic algorithm
    * fintess function is assumed to stay the same during evolution
    * if you want to change the function, feel free to create another object
    """

    def __init__(self):
        # [generation1, generation2, ...]
        self.fitness_history = []
        # optional, comments about how this generation was created
        self.notes = []

    def update_history(self, population: list, note: str = ""):
        fitness_values = [i.fitness.values for i in population]
        self.fitness_history.append(fitness_values)
        self.notes.append(note)

    def get_agg_functions_for_fitness(self):
        means, medians, stds = [], [], []
        for fitness_values in self.fitness_history:
            means.append(np.mean(fitness_values, axis=0))
            medians.append(np.median(fitness_values, axis=0))
            stds.append(np.std(fitness_values, axis=0))
        return np.array(means), np.array(medians), np.array(stds)

    def get_uniqueness_scores(self):
        """
        Calculate uniqueness of fitness values in population:
        how many genomes with the same fitness are in the population
        higher score = more fitness values are unique
        """
        uniqueness_scores = [
            len(set(fitness_values)) / len(fitness_values)
            for fitness_values in self.fitness_history
        ]

        # round values for readability
        uniqueness_scores = [round(score, 4) for score in uniqueness_scores]
        return uniqueness_scores

    def get_generation_shifts(self):
        """
        Calculate how much the population has changed compared to previous generation
        """
        # for keeping len(generation_shifts) == len(dataframe)
        generation_shifts = [0]
        n_generations = len(self.fitness_history)
        for i in range(1, n_generations):
            old_fitness = self.fitness_history[i-1]
            new_fitness = self.fitness_history[i]

            n_fresh_fitness = 0
            for f in new_fitness:
                if f not in old_fitness:
                    n_fresh_fitness += 1
            ratio = n_fresh_fitness / len(new_fitness)
            generation_shifts.append(ratio)

        # round values for readability
        generation_shifts = [round(score, 4) for score in generation_shifts]
        return generation_shifts

    def create_fitness_raw_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.fitness_history)
        df["iteration"] = list(range(len(self.fitness_history)))
        df["notes"] = self.notes
        return df

    def create_fitness_info_df(self) -> pd.DataFrame:
        iteration = list(range(len(self.fitness_history)))
        means, medians, stds = self.get_agg_functions_for_fitness()
        uniqueness_scores = self.get_uniqueness_scores()
        generation_shifts = self.get_generation_shifts()

        df = pd.DataFrame({
            "iteration": iteration,
            "notes": self.notes,
            "uniqueness_scores": uniqueness_scores,
            "generation_shifts": generation_shifts,
        })

        n_fitness_objectives = means.shape[1]
        for i in range(n_fitness_objectives):
            df[f"mean_{i}"] = means[:, i]
            df[f"median_{i}"] = medians[:, i]
            df[f"std_{i}"] = stds[:, i]
        return df

    def write_fitness_raw(self, path=None):
        """Save current fitness raw values to CSV file"""
        if not path:
            return
        try:
            df = self.create_fitness_raw_df()
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"Error occured when trying to write fitness history (raw): {e}")

    def write_fitness_stats(self, path=None):
        """Save current fitness stats to CSV file"""
        if not path:
            return
        try:
            df = self.create_fitness_info_df()
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"Error occured when trying to write fitness history (stats): {e}")
