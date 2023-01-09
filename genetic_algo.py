import evolution_methods as evolution
from formula import Formula


class GeneticAlgorithm:
    def __init__(self, population_size) -> None:
        self.population_size = population_size

        # used methods        
        self.initial_population_method: evolution.InitialPopulationAlgorithm = None
        self.fitness_function: evolution.FitnessFunction = None
        self.selection_method: evolution.SelectionAlgorithm = None
        self.crossover_method: evolution.CrossoverAlgorithm = None
        self.mutation_method: evolution.MutationAlgorithm = None

    def set_initial_population_method(self, method):
        self.initial_population_method = method
        return self

    def set_fitness_function(self, method):
        self.fitness_function = method
        return self

    def set_selection_method(self, method):
        self.selection_method = method
        return self

    def set_crossover_method(self, method):
        self.crossover_method = method
        return self

    def set_mutation_method(self, method):
        self.mutation_method = method
        return self

    def is_final_population(self, configuration, formula: Formula) -> bool:
        # returns True if configuration is final population
        # returns False otherwise
        return self.fitness_function.calculate_fitness(configuration, formula) == 1


    def solve(self, formula: Formula) -> list[int]:
        # implements general genetic algorithm
        # returns best configuration

        # initialize population
        self.population = self.initial_population_method.generate_population(
            population_size=self.population_size,
            genome_size=formula.number_of_variables
            )
        self.population = list(self.population) # type: ignore

        if self.population is None:
            raise Exception("Population is empty")

        # loop until solution is found
        while True:
            # select parents
            parents = self.selection_method.select(self.population, self.fitness_function, formula) # type: ignore
            
            if parents is None:
                raise Exception("No parents selected")

            # crossover parents
            children = self.crossover_method.crossover(parents, self.fitness_function)

            # mutate children
            mutated_children = self.mutation_method.mutate(children)

            self.population = mutated_children

            # evaluate children
            # select best children
            # replace worst individuals in population with best children

            # check if solution is found
            # if yes, return best configuration
            # if no, continue loop

            for configuration in self.population:
                if self.is_final_population(configuration, formula):
                    return configuration
