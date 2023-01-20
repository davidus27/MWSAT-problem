# create a unit test where we check that the solution is correct
# based on a a list of known solutions

import os
from formula import Formula
from genetic_algo import Formula, GeneticAlgorithm
from evolution_methods import *

class OptimalityTester:
    def __init__(self) -> None:
        self.solution_files = self.get_available_solutions()
        self.known_solutions = self.get_correct_solutions()

    def get_available_solutions(self):
        filepath = "data/"

        # get all files in the folder ending with .dat
        files = os.listdir(filepath)
        files = [os.path.join(filepath, file) for file in files]
        files = [file for file in files if file.endswith(".dat")]
        return files

    def get_correct_solution(self, filename: str):
        # solution example:
        # uf20-01000 10282 -1 2 3 -4 5 6 7 8 9 10 -11 12 13 14 -15 16 -17 18 19 -20 0
        # name, weight, correct configuration, 0
        # correct configuration is a list of integers where the sign 
        # indicates if the variable is negated or not
        # and the absolute value indicates the variable index starting from 1
        # 0 indicates the end of the configuration
        # return name, weight, correct configuration

        with open(filename, "r") as file:
            solutions = {}
            for line in file:
                line = line.split(" ")
                name = line[0]
                weight = int(line[1])
                # final configuration will be a list of 0 and 1 where 0 indicates negation
                # on the index of the variable
                configuration = [int(x) for x in line[2:-1]]
                final_configuration = [0] * len(configuration)
                for value in configuration:
                    if value < 0:
                        final_configuration[abs(value) - 1] = 1
                solutions[name] = {"weight": weight, "configuration": final_configuration}

        return solutions
    
    def get_correct_solutions(self):
        solutions = {}
        for filename in self.solution_files:
            solutions[filename] = self.get_correct_solution(filename)
        return solutions


    def buildEvolutionAlgorithm(self):
        return GeneticAlgorithm(population_size=100, reproduction_count=50, new_blood=70, elitism=True, survivors=10, max_iterations=100) \
            .set_initial_population_method(RandomInitialPopulation()) \
            .set_fitness_function(SuccessRateFitnessFunction(strict_satisfiability=True)) \
            .set_selection_method(RouletteSelection()) \
            .set_crossover_method(SinglePointCrossover()) \
            .set_mutation_method(BitFlipMutation(mutation_chance=0.01))
    
    def find_solution(self, filename: str):
        formula = Formula(filename)
        evolution_algorithm = self.buildEvolutionAlgorithm()
        configuration = evolution_algorithm.solve(formula)
        weight = formula.get_total_weight(configuration)
        return configuration, weight

    def test_solution(self, filename):
        # filename = "data/wuf20-71-M/wuf20-01.mwcnf"
        config, solution_weight = self.find_solution(filename)


        # TODO: this it wrong. FIX IT
        path, repo, f  = filename.split('/')
        found_results = self.known_solutions[path + "/" + repo + "-opt.dat"][f[1:-6]]
        
        if not found_results:
            print("No known solution for this file", filename)
            return

        known_weight = found_results["weight"]
        known_configuration = found_results["configuration"]

        # TODO: get array equality
        return known_weight == solution_weight, known_configuration == config