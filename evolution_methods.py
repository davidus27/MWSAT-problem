from abc import ABC, abstractmethod
import random
import math
from formula import Formula

# Initation method


class InitialPopulationAlgorithm:
    def __init__(self) -> None:
        pass

    def __name__(self):
        return "Initial Population Algorithm"

    @abstractmethod
    def generate_population(self, population_size: int, genome_size: int):
        pass


class RandomInitialPopulation(InitialPopulationAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    def __name__(self):
        return "Random Initial Population"

    def generate_population(self, population_size: int, genome_size: int):
        # population can contain either 1 or 0
        yield [ 0 for _ in range(genome_size) ]
        yield [ 1 for _ in range(genome_size) ]
        for _ in range(population_size):
            # generate random individual containing list of 1 and 0
            yield [random.randint(0, 1) for _ in range(genome_size)]

# calculate Fitness function


class FitnessFunction:
    def __init__(self, strict_satisfiability=True) -> None:
        self.strict_satisfiability = strict_satisfiability

    def __name__(self):
        return f"Fitness Function {'strict' if self.strict_satisfiability else 'non-strict'}"

    def calculate_fitness(self, configuration: list[int], formula: Formula) -> float:
        # returns fitness of configuration in formula
        # if strict_satisfiability is True, then configuration must satisfy formula on 100% to get any non-zero fitness
        # otherwise it can satisfy formula on any value between 0 and 1
        if not self.strict_satisfiability or formula.does_satisfy(configuration):
            return self.get_fitness(configuration, formula)
        return 0

    @abstractmethod
    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        pass

# simply look how much is formula satisfied. If on 100% it returns total weight of the formula. Otherwise 0
class SuccessRateFitnessFunction(FitnessFunction):
    def __init__(self, strict_satisfiability=False) -> None:
        super().__init__(strict_satisfiability)
    
    def __name__(self):
        return f"Success Rate Fitness Function {'strict' if self.strict_satisfiability else 'non-strict'}"

    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        return formula.get_total_weight(configuration)

class PrioritizedFitnessFunction(FitnessFunction):
    def __init__(self, weight_priority=0.2, satisfiability_priority=0.8) -> None:
        super().__init__(False)
        self.weight_priority = weight_priority
        self.satisfiability_priority = satisfiability_priority

    def __name__(self):
        return f"Prioritized Fitness Function {self.weight_priority} {self.satisfiability_priority}"

    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        weight = formula.get_total_weight(configuration)
        satisfiability = formula.get_success_rate(configuration)
        return weight * self.weight_priority + satisfiability * self.satisfiability_priority

# map success rate between 0 and 1
# multiply it by weight of current configuration
class MappedPunishedFitnessFunction(FitnessFunction):
    def __init__(self, strict_satisfiability=False) -> None:
        super().__init__(strict_satisfiability)

    def __name__(self):
        return f"Mapped Punished Fitness Function {'strict' if self.strict_satisfiability else 'non-strict'}"

    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        mapped_value = formula.get_mapped_weight(configuration)
        success_rate = formula.get_success_rate(configuration)
        return mapped_value * success_rate

# map success rate between 0 and 1 again
# now the success rate will be also used in logistic function
class LogisticFitnessFunction(FitnessFunction):
    def __init__(self, strict_satisfiability=False, steepness_param=100, midpoint_curve=0.99) -> None:
        super().__init__(strict_satisfiability)
        self.steepness_param = steepness_param
        self.midpoint_curve = midpoint_curve

    def __name__(self):
        return f"Logistic Fitness Function {'strict' if self.strict_satisfiability else 'non-strict'} {self.steepness_param} {self.midpoint_curve}"

    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        mapped_value = formula.get_mapped_weight(configuration)
        success_rate = formula.get_success_rate(configuration)

        # Use the logistic function to map the success rate to the range 0 to 1
        success_ratio = 1 / (1 + math.exp(-self.steepness_param * (success_rate - self.midpoint_curve)))

        return mapped_value * success_ratio


class ExponentialFitnessFunction(FitnessFunction):
    def __init__(self, strict_satisfiability=False, steepness_param=2, midpoint_curve=0.5, promoting_factor=0.99) -> None:
        super().__init__(strict_satisfiability)
        self.steepness_param = steepness_param
        self.midpoint_curve = midpoint_curve
        self.promoting_factor = promoting_factor

    def __name__(self):
        return f"Exponential Fitness Function {'strict' if self.strict_satisfiability else 'non-strict'} {self.steepness_param} {self.midpoint_curve} {self.promoting_factor}"

    def get_fitness(self, configuration: list[int], formula: Formula) -> float:
        satisfiability = formula.get_success_rate(configuration)
        # mapped_weight = formula.get_mapped_weight(configuration)
        
        # prioritize satisfiability using hyperbolic function
        return math.exp(satisfiability * self.steepness_param)

# Selection phase
class SelectionAlgorithm:
    def __init__(self) -> None:
        self.parent_count = 2

    def __name__(self):
        return f"Selection Algorithm {self.parent_count}"

    @abstractmethod
    def select(self, population: list, fitness_function, formula: Formula):
        pass


class RouletteSelection(SelectionAlgorithm):
    def __init__(self):
        super().__init__()

    def __name__(self):
        return "Roulette Selection"

    def select(self, population: list, fitness_function, formula: Formula) -> list:
        # implement roulette selection
        # return two parents

        # calculate total fitness
        fitness_values = []
        total_fitness = 0
        for individual in population:
            fitness_values.append(fitness_function.calculate_fitness(
                individual, formula))
            total_fitness += fitness_values[-1]

        # calculate total fitness faster

        # select two parents
        parents = []
        for _ in range(self.parent_count):

            # select parent when all total fitness is zero
            if total_fitness == 0:
                parents.append(random.choice(population))
                continue

            # select random number
            random_number = random.uniform(0, total_fitness)

            # select parent
            for individual_index, individual in enumerate(population):
                # random_number -= fitness_function.calculate_fitness(
                #     individual, formula)
                random_number -= fitness_values[individual_index]
                
                if random_number <= 0:
                    parents.append(individual)
                    break

        return parents

class TournamentSelection(SelectionAlgorithm):
    def __init__(self, tournament_size=2):
        super().__init__()
        self.tournament_size = tournament_size
    
    def __name__(self):
        return f"Tournament Selection {self.tournament_size}"

    def select(self, population: list, fitness_function, formula: Formula) -> list:
        # implement tournament selection
        # return two parents

        # select two parents
        parents = []
        for _ in range(self.parent_count):
            # select tournament
            tournament = random.choices(population, k=self.tournament_size)

            # select best parent
            best_parent = tournament[0]
            best_parent_fitness = fitness_function.calculate_fitness(
                best_parent, formula)
            for individual in tournament[1:]:
                individual_fitness = fitness_function.calculate_fitness(
                    individual, formula)
                if individual_fitness > best_parent_fitness:
                    best_parent = individual
                    best_parent_fitness = individual_fitness

            parents.append(best_parent)

        return parents

class RankSelection(SelectionAlgorithm):
    def __init__(self):
        super().__init__()

    def __name__(self):
        return f"Rank Selection {self.parent_count}"

    def select(self, population: list, fitness_function, formula: Formula) -> list:
        # implement rank selection
        # return two parents

        # calculate total fitness
        fitness_values = []
        total_fitness = 0
        for individual in population:
            fitness_values.append(fitness_function.calculate_fitness(
                individual, formula))
            total_fitness += fitness_values[-1]

        # calculate total fitness faster

        # select two parents
        parents = []
        for _ in range(self.parent_count):

            # select parent when all total fitness is zero
            if total_fitness == 0:
                parents.append(random.choice(population))
                continue

            # select random number
            random_number = random.uniform(0, total_fitness)

            # select parent
            for individual_index, individual in enumerate(population):
                # random_number -= fitness_function.calculate_fitness(
                #     individual, formula)
                random_number -= fitness_values[individual_index]
                
                if random_number <= 0:
                    parents.append(individual)
                    break

        return parents


class BoltzmannSelection(SelectionAlgorithm):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def __name__(self):
        return f"Boltzmann Selection {self.temperature}"

    def select(self, population: list, fitness_function, formula: Formula) -> list:
        # implement boltzmann selection
        # return two parents

        # calculate total fitness
        fitness_values = []
        total_fitness = 0
        for individual in population:
            fitness_values.append(fitness_function.calculate_fitness(
                individual, formula))
            total_fitness += fitness_values[-1]

        # calculate total fitness faster

        # select two parents
        parents = []
        for _ in range(self.parent_count):

            # select parent when all total fitness is zero
            if total_fitness == 0:
                parents.append(random.choice(population))
                continue

            # select random number
            random_number = random.uniform(0, total_fitness)

            # select parent
            for individual_index, individual in enumerate(population):
                # random_number -= fitness_function.calculate_fitness(
                #     individual, formula)
                random_number -= fitness_values[individual_index]
                
                if random_number <= 0:
                    parents.append(individual)
                    break

        return parents

class StochasticUniversalSamplingSelection(SelectionAlgorithm):
    def __init__(self):
        super().__init__()

    def __name__(self):
        return f"Stochastic Universal Sampling Selection {self.parent_count}"

    def select(self, population: list, fitness_function, formula: Formula) -> list:
        # implement stochastic universal sampling selection
        # return two parents

        # calculate total fitness
        fitness_values = []
        total_fitness = 0
        for individual in population:
            fitness_values.append(fitness_function.calculate_fitness(
                individual, formula))
            total_fitness += fitness_values[-1]

        # calculate total fitness faster

        # select two parents
        parents = []
        for _ in range(self.parent_count):

            # select parent when all total fitness is zero
            if total_fitness == 0:
                parents.append(random.choice(population))
                continue

            # select random number
            random_number = random.uniform(0, total_fitness)

            # select parent
            for individual_index, individual in enumerate(population):
                # random_number -= fitness_function.calculate_fitness(
                #     individual, formula)
                random_number -= fitness_values[individual_index]
                
                if random_number <= 0:
                    parents.append(individual)
                    break

        return parents

# Crossover phase

class CrossoverAlgorithm:
    def __init__(self, children_count=2) -> None:
        self.children_count = children_count

    def __name__(self):
        return f"Crossover Algorithm {self.children_count}"

    @abstractmethod
    def crossover(self, parents: list) -> list:
        pass


class SinglePointCrossover(CrossoverAlgorithm):
    def __init__(self, children_count=2) -> None:
        super().__init__(children_count)

    def __name__(self):
        return f"Single Point Crossover {self.children_count}"

    def crossover(self, parents):
        assert len(parents) == 2 # TODO: make it for multiple parents
        
        parent1, parent2 = parents
        children = []
        for _ in range(self.children_count):
            # Choose a crossover point at random
            crossover_point = random.randint(0, len(parent1))

            # Create the child by combining the two parents at the crossover point
            child = parent1[:crossover_point] + parent2[crossover_point:]
            children.append(child)

            # Swap parents
            parent1, parent2 = parent2, parent1

        return children



class KPointCrossover(CrossoverAlgorithm):
    def __init__(self, children_count=2, k_points=2) -> None:
        super().__init__(children_count)
        self.k_points = k_points

    def __name__(self):
        return f"K Point Crossover {self.children_count} {self.k_points}"

    def crossover(self, parents: list):
        assert len(parents) == 2 # TODO: Maybe make it for more parents

        parent1, parent2 = parents
        children = []
        for _ in range(self.children_count):
            # Choose k crossover points at random
            crossover_points = sorted(random.sample(range(len(parent1)), self.k_points))
            crossover_points.insert(0, -1)
            crossover_points.append(len(parent1))

            # Create the child by combining the two parents at the crossover points
            child = []
            for j in range(self.k_points + 1):
                child += parent1[crossover_points[j]+1:crossover_points[j+1]]
                child += parent2[crossover_points[j]+1:crossover_points[j+1]]

            children.append(child)

            # Swap parents
            parent1, parent2 = parent2, parent1

        return children

class UniformCrossover(CrossoverAlgorithm):
    def __init__(self, children_count=2, parents_count=2) -> None:
        super().__init__(children_count)
        self.parents_count = parents_count
        self.child_genome_picker = None

    def __name__(self):
        return f"Uniform Crossover {self.children_count} {self.parents_count}"

    def crossover(self, parents: list):
        # generate random vector of 0 to parent_count - 1
        if self.child_genome_picker is None:
            self.child_genome_picker = [random.randint(0, self.parents_count - 1) for _ in range(len(parents[0]))]

        genome_length = len(parents[0])
        children = []
        for _ in range(self.children_count):

            child = [0] * genome_length
            for current_index, parent_index in enumerate(self.child_genome_picker):
                child[current_index] = parents[parent_index][current_index]

            children.append(child)

        return children

# Mutation phase

class MutationAlgorithm:
    def __init__(self) -> None:
        pass

    def __name__(self):
        return "Mutation Algorithm"

    @abstractmethod
    def mutate(self, population):
        pass


class BitFlipMutation(MutationAlgorithm):
    def __init__(self, mutation_chance=0.01) -> None:
        self.mutation_chance = mutation_chance
        super().__init__()

    def __name__(self):
        return f"Bit Flip Mutation {self.mutation_chance}"

    def mutate_individual(self, individual):
        # choose random bit in gene
        genome_index = random.randint(0, len(individual) - 1)
        # flip bit
        individual[genome_index] = 1 - individual[genome_index]
        return individual

    def mutate(self, population):
        for index, individual in enumerate(population):
            if random.random() < self.mutation_chance:
                population[index] = self.mutate_individual(individual) 
        return population
