# general
import numpy as np
# population
from sampo.hybrid.population import PopulationScheduler
from sampo.api.genetic_api import Individual
# schemas
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas import WorkGraph
# my stuff
from my_modules.precedence import PrecedenceManager
from my_modules.project_info import ProjectInfo
# for fitness
from sampo.scheduler.genetic.operators import TimeFitness, DeadlineCostFitness, TimeAndResourcesFitness, TimeAndCostFitness


class Solution:

    def __init__(
            self, 
            id_: int, 
            activity_list: np.array, 
            resources_matrix: np.array, 
            ceil_list: np.array, 
            fitness: tuple|None = None
        ):
        
        self.id_ = id_
        self.activity_list = np.array(activity_list).copy()
        self.resources_matrix = np.array(resources_matrix).copy()
        self.ceil_list = np.array(ceil_list).copy()
        self.fitness = fitness

        
    @classmethod
    def from_individual(cls, id_: int, individual: list):
        return cls(
            id_=id_,
            activity_list=individual[0], 
            resources_matrix=individual[1], 
            ceil_list=individual[2], 
            fitness=individual.fitness.values
        )


class Population:

    def __init__(self, solutions: list[Solution], wg, contractors, fitness_function):
        self.solutions = solutions
        self.population_size = len(solutions)
        
        self.project_info = ProjectInfo(wg, contractors)
        self.precedence_manager = PrecedenceManager(self.project_info.children, self.project_info.parents)
        self.fitness_function = fitness_function
        
    
    @classmethod
    def from_population(cls, population: list, wg, contractors, fitness_function):
        solutions = [Solution.from_individual(id_, i) for id_, i in enumerate(population)]
        return cls(solutions, wg, contractors, fitness_function) 

    
    def to_population(self):
        individuals = [
            Individual(
                individual_fitness_constructor=TimeFitness,
                chromosome=[solution.activity_list, solution.resources_matrix, solution.ceil_list, ScheduleSpec(), np.array([])]
            )
            for solution in self.solutions
        ]
        for i in range(len(self.solutions)):
            individuals[i].fitness.values = self.solutions[i].fitness

        return individuals

    
    def mutation(self, solution):

        new_activity_list = solution.activity_list.copy()
        new_resources_matrix = solution.resources_matrix.copy()
        
        n_resource_mutations = np.random.binomial(len(self.project_info.nonzero_resources_indices), 0.02)
        for _ in range(n_resource_mutations):
            n_nonzero_resources = len(self.project_info.nonzero_resources_indices)
            work_to_mutate, worker_to_mutate = self.project_info.nonzero_resources_indices[np.random.choice(n_nonzero_resources)]
            min_amount, max_amount = self.project_info.get_resource_borders(work_id=work_to_mutate, worker_id=worker_to_mutate)
            new_resources_matrix[work_to_mutate, worker_to_mutate] = np.random.randint(min_amount, max_amount+1)
    
        n_order_mutations = np.random.binomial(self.project_info.n_works, 0.02)
        for _ in range(n_order_mutations):
            a, b = np.random.randint(self.project_info.n_works, size=2)
            new_activity_list[[a, b]] = new_activity_list[[b, a]]
        new_activity_list = self.precedence_manager.convert_al_to_valid_order(new_activity_list)

        return Solution(solution.id_, new_activity_list, new_resources_matrix, solution.ceil_list.copy(), fitness=None)


    def crossover(self, solution_1, solution_2, inherit_from_first=0.90):
        # TODO how to do fast valid crossover for solutions?
        new_activity_list = solution_1.activity_list.copy()
    
        new_resources_matrix = np.where(
            np.random.choice([0, 1], p=[1-inherit_from_first, inherit_from_first], size=(self.project_info.n_works, self.project_info.n_workers+1)),
            solution_1.resources_matrix,
            solution_2.resources_matrix
        )

        return Solution(solution_1.id_, new_activity_list, new_resources_matrix, solution_1.ceil_list.copy(), fitness=None)


    @staticmethod
    def is_multi_fitness_better(a, b):
        if len(a) == len(b) == 1:
            return a < b
        return ((a[0] < b[0]) and (a[1] <= b[1])) or ((a[0] <= b[0]) and (a[1] < b[1])) 








class BeeColony(Population):

    def update(self):
        # get new solutions
        new_solutions = []

        # random direction solutions
        for _ in range(self.population_size):
            solution_index = np.random.choice(self.population_size)
            new_solutions.append( self.mutation(self.solutions[solution_index]) )
        
        # toward other solutions
        for _ in range(self.population_size):
            solution_1_index = np.random.choice(self.population_size)
            solution_2_index = np.random.choice(self.population_size)
            new_solutions.append(
                self.crossover(
                    self.solutions[solution_1_index],
                    self.solutions[solution_2_index],
                    inherit_from_first=0.90,
                )
            )
        
        # calculate and set fitness
        new_fitness = self.fitness_function(new_solutions)
        for i in range(len(new_solutions)):
            new_solutions[i].fitness = new_fitness[i]

        # update positions
        for i in range(len(new_solutions)):
            # if improve: move
            # if new_solutions[i].fitness < self.solutions[new_solutions[i].id_].fitness:
            if self.is_multi_fitness_better(new_solutions[i].fitness, self.solutions[new_solutions[i].id_].fitness):
                self.solutions[new_solutions[i].id_] = new_solutions[i]



class ArtificialImmune(Population):

    def update(self):

        affinity = [solution.fitness[0] for solution in self.solutions]
        affinity = (-1) * np.array(affinity)
        affinity = (affinity - np.min(affinity)) / (np.max(affinity) - np.min(affinity))
        # number of clones
        n_clones = np.ceil(affinity * 4).astype(np.int16)
        
        # get new solutions
        new_solutions = []

        # add clones
        for solution_index in range(self.population_size):
            for clone_number in range(n_clones[solution_index]):
                new_solutions.append( self.mutation(self.solutions[solution_index]) )

        # calculate and set fitness
        new_fitness = self.fitness_function(new_solutions)
        for i in range(len(new_solutions)):
            new_solutions[i].fitness = new_fitness[i]
        
        all_solutions = sorted(self.solutions + new_solutions, key=lambda x: x.fitness)
        n_chosen_best = round(self.population_size * 0.8)
        
        self.solutions = all_solutions[:n_chosen_best] + list(np.random.choice(
            all_solutions[n_chosen_best:], 
            size=self.population_size-n_chosen_best, 
            replace=False
        ))

















class RandomScheduler(PopulationScheduler):
    
    def __init__(self, wg, contractors, fitness_function, population_size=50):
        self.wg = wg
        self.contractors = contractors
        self.fitness_function = fitness_function
        self.population_size = population_size
        
        self.project_info = ProjectInfo(wg, contractors)
        self.precedence_manager = PrecedenceManager(self.project_info.children, self.project_info.parents)

    
    def schedule(self,
                 initial_population: list,
                 wg: WorkGraph,
                 contractors: list,
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list:

        if initial_population != []:
            raise Exception("RandomScheduler should be used only for initialization, not updating")
        
        # generate initial population, randomly
        solutions = []
        for i in range(self.population_size):
            activity_list = self.precedence_manager.convert_al_to_valid_order(np.random.permutation(self.project_info.n_works))
            activity_list = activity_list.astype(np.int16)
            
            new_resources_matrix = np.zeros((self.project_info.n_works, self.project_info.n_workers+1))
            for work_id in range(self.project_info.n_works):
                for worker_id in range(self.project_info.n_workers):
                    min_amount, max_amount = self.project_info.get_resource_borders(work_id=work_id, worker_id=worker_id)
                    new_resources_matrix[work_id, worker_id] = np.random.randint(min_amount, max_amount+1)
            new_resources_matrix = new_resources_matrix.astype(np.int16)
            
            ceil_list = self.project_info.contractor_borders.copy()
            
            solutions.append(Solution(i, activity_list, new_resources_matrix, ceil_list))
        
        fitness_values = self.fitness_function(solutions)
        for i in range(len(solutions)):
            solutions[i].fitness = fitness_values[i]

        return Population(solutions, wg, contractors, self.fitness_function).to_population()

































class Bee(Solution):

    def __init__(
            self, 
            id_: int, 
            activity_list: np.array, 
            resources_matrix: np.array, 
            ceil_list: np.array, 
            fitness: tuple|None = None,
            n_tries: int = 0
        ):
        super().__init__(id_, activity_list, resources_matrix, ceil_list, fitness)
        self.n_tries = n_tries
        


class BeeScheduler(PopulationScheduler):
    def __init__(self, wg, contractors, fitness_function):
        self.wg = wg
        self.contractors = contractors
        self.fitness_function = fitness_function
        
        self.project_info = ProjectInfo(wg, contractors)
        self.precedence_manager = PrecedenceManager(self.project_info.children, self.project_info.parents)

    
    def schedule(self,
                 initial_population: list,
                 wg: WorkGraph,
                 contractors: list,
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list:
    
        colony = BeeColony.from_population(initial_population, wg, contractors, self.fitness_function)
        for i in range(1):
            colony.update()
        population = colony.to_population()
        return population


class ImmuneScheduler(PopulationScheduler):
    
    def __init__(self, wg, contractors, fitness_function):
        self.wg = wg
        self.contractors = contractors
        self.fitness_function = fitness_function
        
        self.project_info = ProjectInfo(wg, contractors)
        self.precedence_manager = PrecedenceManager(self.project_info.children, self.project_info.parents)

    
    def schedule(self,
                 initial_population: list,
                 wg: WorkGraph,
                 contractors: list,
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list:
    
        colony = ArtificialImmune.from_population(initial_population, wg, contractors, self.fitness_function)
        for i in range(1):
            colony.update()
        population = colony.to_population()
        return population










        