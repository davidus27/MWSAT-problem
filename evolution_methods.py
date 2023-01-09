from abc import ABC, abstractmethod
import random
from formula import Formula

# Initation method
class InitialPopulationAlgorithm:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_population(self, population_size: int, genome_size: int):
        pass

class RandomInitialPopulation(InitialPopulationAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    def generate_population(self, population_size: int, genome_size: int):
        # population can contain either 1 or 0
        for _ in range(population_size):
            # generate random individual containing list of 1 and 0
            yield [ random.randint(0, 1) for _ in range(genome_size) ]

# calculate Fitness function
class FitnessFunction:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate_fitness(self, configuration: list[int], formula: Formula) -> float:
        pass

class SuccessRateFitnessFunction(FitnessFunction):
    def __init__(self) -> None:
        super().__init__()

    def calculate_fitness(self, configuration: list[int], formula: Formula) -> float:
        return formula.get_success_rate(configuration)

class PunishedSuccessRateFitnessFunction(FitnessFunction):
    
    def calculate_fitness(self, configuration: list[int], formula: Formula) -> float:
        fitness = 0
        results = formula.get_clauses_results(configuration)
        for clause_index, clause_result in enumerate(results):
            if clause_result == 1:
                fitness += formula.get_clause_weight_by_index(clause_index, configuration)
            else:
                fitness -= formula.get_clause_weight_by_index(clause_index, configuration)
        
        return fitness


# Selection phase

class SelectionAlgorithm:
    def __init__(self) -> None:
        self.parent_count = 2

    @abstractmethod
    def select(self, population: list, fitness_function, formula: Formula):
        pass

class RouletteSelection(SelectionAlgorithm):
    def __init__(self):
        super().__init__()
    
    def select(self, population: list, fitness_function, formula: Formula):
        # implement roulette selection
        # return two parents

        # calculate total fitness
        total_fitness = 0
        for individual in population:
            total_fitness += fitness_function.calculate_fitness(individual, formula)

        # select two parents
        parents = []
        for _ in range(self.parent_count):
            # select random number
            random_number = random.uniform(0, total_fitness)

            # select parent
            for individual in population:
                random_number -= fitness_function.calculate_fitness(individual, formula)
                if random_number <= 0:
                    parents.append(individual)
                    break
        
        return parents


# Crossover phase

class CrossoverAlgorithm:
    def __init__(self, parent_count=2, children_count=2) -> None:
        self.parent_count = parent_count
        self.children_count = children_count
    
    @abstractmethod
    def crossover(self, population: list, fitness_function):
        pass


class SinglePointCrossover(CrossoverAlgorithm):
    def __init__(self):
        super().__init__()
    
    def crossover(self, population: list, fitness_function):
        # select two parents
        # perform crossover
        # return two children
        
        # select random crossover point
        crossover_point = random.randint(0, len(population[0]) - 1)

        # perform crossover
        children = []
        

class TwoPointCrossover(CrossoverAlgorithm):
    def __init__(self):
        super().__init__()
    
    def crossover(self, pupulation: list, fitness_function):
        # select two parents
        # perform crossover
        # return two children
        pass

# Mutation phase

class MutationAlgorithm:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def mutate(self, population):
        pass

class BitFlipMutation(MutationAlgorithm):
    def __init__(self, mutation_chance=0.01) -> None:
        self.mutation_chance = mutation_chance
        super().__init__()


    def mutate(self, population: list):
        # flip one bit
        if random.random() < self.mutation_chance:
            individual_index = random.randint(0, len(population) - 1)
            genome_index = random.randint(0, len(population[individual_index]) - 1)
            population[individual_index][genome_index] = 1 - population[individual_index][genome_index]
        return population