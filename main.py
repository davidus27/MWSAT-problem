import os
import random
from genetic_algo import Formula, GeneticAlgorithm
from evolution_methods import *


def create_random_config(number_of_variables):
    config = []
    for _ in range(number_of_variables):
        if random.random() < 0.5:
            config.append(1)
        else:
            config.append(0)
    return config


def buildEvolutionAlgorithm():
    return GeneticAlgorithm(population_size=50, reproduction_count=10, new_blood=30, elitism=True, survivors=10, max_iterations=100) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(SuccessRateFitnessFunction()) \
        .set_selection_method(RouletteSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=0.1))


def solve_for_file(filename: str):
    formula = Formula(filename)

    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()
    solution = evolution_algorithm.solve(formula)

    print("Solution: ", solution)
    print("Weight: ", formula.get_total_weight(solution))

    return solution


def main():
    filename = os.path.join(os.getcwd(), "data",
                            "wuf20-71-M", "wuf20-01.mwcnf")

    solve_for_file(filename)


if __name__ == "__main__":
    main()
