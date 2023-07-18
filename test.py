# create a unit test where we check that the solution is correct
# based on a a list of known solutions

import os
import sys
from formula import Formula
from genetic_algo import Formula, GeneticAlgorithm
from evolution_methods import *
import threading

# create a function to build the evolution algorithm
# with parameters that we can change
def buildEvolutionAlgorithm(population_size=500, reproduction_count=250, new_blood=80, elite_size=30, survivors=5, max_iterations=350, mutation_chance=0.01):
    return GeneticAlgorithm(population_size=population_size, reproduction_count=reproduction_count, new_blood=new_blood, elite_size=elite_size, survivors=survivors, max_iterations=max_iterations) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(LogisticFitnessFunction(strict_satisfiability=False)) \
        .set_selection_method(TournamentSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=mutation_chance))



def solve(formula: Formula, evolution_algorithm):
    solution = evolution_algorithm.solve(formula)
    success_rate = formula.get_success_rate(solution)
    total_weight = formula.get_total_weight(solution)
    perfomance = evolution_algorithm.get_perfomance_tracker()
    evolution_algorithm.reset_perfomance_tracker()

    return {
        "filename": formula.filename,
        "method": evolution_algorithm.get_used_method(),
        "solution": solution,
        "success_rate": success_rate,
        "total_weight": total_weight,
        "perfomances": perfomance,
        "success": success_rate,
    }

def generate_evolution_settings():
    # this will generate a list of settings
    # that we can use to test the evolution algorithm
    # with different settings
    settings = []
    for population_size in [100, 500]:
        for reproduction_count in [100, 200, 250]:
            for new_blood in [0, 100]:
                for elite_size in [0, 10, 50]:
                    for survivors in [0, 10]:
                        for max_iterations in [500]:
                            for mutation_chance in [0, 0.01, 0.1, 0.2]:
                                yield ({
                                    "population_size": population_size,
                                    "reproduction_count": reproduction_count,
                                    "new_blood": new_blood,
                                    "elite_size": elite_size,
                                    "survivors": survivors,
                                    "max_iterations": max_iterations,
                                    "mutation_chance": mutation_chance,
                                })



def generate_evolution_algorithms():
    # this will generate a list of evolution algorithms
    # with different settings

    fitness_functions = [
        SuccessRateFitnessFunction(strict_satisfiability=True),
        LogisticFitnessFunction(strict_satisfiability=False),
        PrioritizedFitnessFunction(weight_priority=0.2, satisfiability_priority=0.8),
        ExponentialFitnessFunction(strict_satisfiability=False),
    ]

    selection_methods = [
        TournamentSelection(),
        RouletteSelection(),
        RankSelection(),
        BoltzmannSelection(),
        StochasticUniversalSamplingSelection(),
    ]

    crossover_methods = [
        SinglePointCrossover(),
        UniformCrossover(),
    ]

    algorithms = []

    # First we will test the evolution algorithm with different sizes
    for setting in generate_evolution_settings():
        algorithms.append(
            GeneticAlgorithm(setting["population_size"], setting["reproduction_count"], setting["new_blood"], setting["elite_size"], setting["survivors"], setting["max_iterations"]) \
                .set_initial_population_method(RandomInitialPopulation()) \
                .set_fitness_function(SuccessRateFitnessFunction(strict_satisfiability=False)) \
                .set_selection_method(TournamentSelection()) \
                .set_crossover_method(SinglePointCrossover()) \
                .set_mutation_method(BitFlipMutation(mutation_chance=setting["mutation_chance"]))
        )

    # Then we will test the evolution algorithm with different functions
    for fitness_function in fitness_functions:
        for selection_method in selection_methods:
            for crossover_method in crossover_methods:
                algorithms.append(
                    GeneticAlgorithm(population_size=500, reproduction_count=250, new_blood=80, elite_size=30, survivors=10, max_iterations=350) \
                        .set_initial_population_method(RandomInitialPopulation()) \
                        .set_fitness_function(fitness_function) \
                        .set_selection_method(selection_method) \
                        .set_crossover_method(crossover_method) \
                        .set_mutation_method(BitFlipMutation(mutation_chance=0.01))
                )

    return algorithms
    

def test_one_folder(directory, output_file):
    # this will test the evolution algorithm
    # with different settings and functions
    # and save the results to a file

    algorithms = generate_evolution_algorithms()

    with open(output_file, "w") as file:
        for filename in os.listdir(directory)[:10]:
            for algorithm in algorithms:
                if not filename.endswith(".mwcnf"):
                    continue
                formula = Formula(directory + "/" + filename)
                result = solve(formula, algorithm)
                
                file.write(str(result))
                file.write("\n")



def test_all(directories):
    # create a thread for each directory
    threads = []
    for directory in directories:
        base_name = os.path.basename(directory)
        output_file = "white_box/" + base_name + ".txt"

        threads.append(threading.Thread(target=test_one_folder, args=(directory, output_file)))
    
    # start all threads
    for thread in threads:
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # run the test on multiple folders in data/ on multiple threads
    # and save the results to a file
    directories = ["data/wuf50-218-R", "data/wuf50-218R-R"]
    test_all(directories)