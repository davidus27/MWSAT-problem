import evolution_methods as evolution
from formula import Formula
import random



class GeneticAlgorithm:
    def __init__(self, population_size: int, reproduction_count: int, new_blood=0, elite_size=1, survivors=0, max_iterations=100) -> None:
        self.population_size = population_size
        self.reproduction_count = reproduction_count
        self.new_blood = new_blood
        self.elitism = elite_size
        self.survivors = survivors
        self.max_iterations = max_iterations
        
        self.perfomance_tracker = []

        assert self.population_size > 0
        assert self.reproduction_count > 0
        # assert self.new_blood <= self.population_size

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

    def __printout(self, formula: Formula):
        # find me currently best in population
        # We want to find the best configuration that has the smallest amount of variables set to 1 and F(Y) = 1
        assert self.population != None
        currently_best = sorted(self.population, key=lambda x: self.fitness_function.calculate_fitness(
                x, formula), reverse=True)[0]
        print("The best fitness:", self.fitness_function.calculate_fitness(currently_best, formula))
        print("The best configuration success rate:", formula.get_success_rate(currently_best))


    def __track_performance(self, best, formula: Formula):
        # find me currently best in population
        # We want to find the best configuration that has the smallest amount of variables set to 1 and F(Y) = 1
        self.perfomance_tracker.append(self.fitness_function.calculate_fitness(best, formula))

    def get_perfomance_tracker(self):
        return self.perfomance_tracker


    def solve(self, formula: Formula) -> list[int]:
        # implements general genetic algorithm
        # returns best configuration

        # initialize population
        self.population = self.initial_population_method.generate_population(
            population_size=self.population_size,
            genome_size=formula.number_of_variables
        )
        self.population = list(self.population)  # type: ignore

        # loop until solution is found
        for iteration_count in range(self.max_iterations):
            
            # print("Iteration:", iteration_count)
            # self.__printout(formula)

            new_generation = []

            assert self.population != None
            for _ in range(self.reproduction_count):
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)

                child = self.crossover_method.crossover([parent1, parent2])
                
                # mutate children
                children = self.mutation_method.mutate(child)
                
                assert children != None
                new_generation.extend(children)

            self.population.sort(key=lambda x: self.fitness_function.calculate_fitness(x, formula), reverse=True)

            best = self.population[0]

            if self.elitism > 0:
                assert self.population != None
                new_generation.extend(self.population[:self.elitism])

            # if formula.get_success_rate(best) == 1.0:
            #     print("Solution found i:", iteration_count)
            #     return best

            # print("Currently best:", formula.get_success_rate(best))

            if self.new_blood > 0:
                # generate new individuals
                new_blood = self.initial_population_method.generate_population(
                    population_size=self.new_blood,
                    genome_size=formula.number_of_variables
                )
                new_blood = list(new_blood)  # type: ignore
                new_generation += new_blood

            if self.survivors > 0:
                assert self.population != None
                # random survivors
                survivors = random.sample(self.population, self.survivors)
                new_generation += survivors

            self.population = new_generation
                        
            # track performance
            self.__track_performance(self.population[0], formula)


        # return the best one from current population
        # type: ignore
        assert self.population != None
        return max(self.population, key=lambda x: self.fitness_function.calculate_fitness(x, formula))
