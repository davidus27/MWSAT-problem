import os
import sys
from genetic_algo import Formula, GeneticAlgorithm
from evolution_methods import *

# profiler 
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


def buildEvolutionAlgorithm():
    return GeneticAlgorithm(population_size=100, reproduction_count=20, new_blood=70, elitism=True, survivors=20, max_iterations=50) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(SuccessRateFitnessFunction()) \
        .set_selection_method(RouletteSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=0.1))


def solve_for_file(filename: str, evolution_algorithm: GeneticAlgorithm):
    formula = Formula(filename)

    solution = evolution_algorithm.solve(formula)

    print("Solution:", solution)
    print("Success rate:", formula.get_success_rate(solution))
    print("Weight:", formula.get_total_weight(solution))

    return solution


def main1():
    if len(sys.argv) > 2:
        print("Usage: python3 main.py <data_folder>")
        return
    if len(sys.argv) == 1:
        file_path = os.path.join(os.getcwd(), "data", "wuf20-71-M")
    else:
        file_path = os.path.join(os.getcwd(), sys.argv[1])

    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()

    for filename in os.listdir(file_path):
        print("Solving file:", filename)
        filename = os.path.join(os.getcwd(), "data", "wuf20-71-M", filename)
        solve_for_file(filename, evolution_algorithm)

def build3():
    return GeneticAlgorithm(population_size=100, reproduction_count=50, new_blood=70, elitism=False, survivors=10, max_iterations=50) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(SuccessRateFitnessFunction()) \
        .set_selection_method(RouletteSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=0.1))


def main():
    filename = "data/wuf100-430-Q/wuf100-01.mwcnf"

    formula = Formula(filename)
    # create evolution algorithm
    evolution_algorithm = build3()
    solution = evolution_algorithm.solve(formula)

    print("Solution:", solution)
    print("Success rate:", formula.get_success_rate(solution))
    print("Weight:", formula.get_total_weight(solution))

if __name__ == "__main__":
    main()
    # with PyCallGraph(output=GraphvizOutput()):
    #     main()

