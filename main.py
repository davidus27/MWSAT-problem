import os
import sys
from genetic_algo import Formula, GeneticAlgorithm
from evolution_methods import *

# create threads
import threading

# profiler 
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput


def buildEvolutionAlgorithm():
    return GeneticAlgorithm(population_size=100, reproduction_count=50, new_blood=70, elitism=True, survivors=10, max_iterations=100) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(SuccessRateFitnessFunction(strict_satisfiability=True)) \
        .set_selection_method(RouletteSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=0.01))


def solve_for_file(filename: str, evolution_algorithm: GeneticAlgorithm):
    formula = Formula(filename)

    solution = evolution_algorithm.solve(formula)
    success_rate = formula.get_success_rate(solution)
    total_weight = formula.get_total_weight(solution)
    
    print("Solving file:", filename)
    print("Solution:", solution)
    print("Success rate:", success_rate)
    print("Weight:", total_weight)

    return solution


def execute_one_file(filename: str, output_file):
    formula = Formula(filename)
    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()
    print("Solving file:", filename)
    # send output to file
    sys.stdout = output_file

    solution = evolution_algorithm.solve(formula)

    print("Solution:", solution)
    print("Success rate:", formula.get_success_rate(solution))
    print("Weight:", formula.get_total_weight(solution))

def thread_function(files: list[str]):
    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()

    for filename in files:
        solve_for_file(filename, evolution_algorithm)

def execute_threading(folder_name: str, output_file, num_threads=4):
    # separate files into num_threads groups
    files = os.listdir(folder_name)
    files = [os.path.join(folder_name, file) for file in files]
    files = [files[i::num_threads] for i in range(num_threads)]

    threads = []
    for i in range(num_threads):
        x = threading.Thread(target=thread_function, args=(files[i],))
        threads.append(x)
        print("Starting thread", i)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()
        print("Thread", index, "done")
    
    print("Done")

def run_all():
    # run threaded for all directories in data

    # get all directories in data that starts with wuf
    data_dir = "data"
    directories = os.listdir(data_dir)
    directories = [directory for directory in directories if directory.startswith("wuf")]
    directories = [os.path.join(data_dir, directory) for directory in directories]
    directories = [directory for directory in directories if os.path.isdir(directory)]

    for directory in directories:
        # create output file
        basename = os.path.basename(directory)
        output_file = open(os.path.join("results", basename), "w")
        # redirect output to file
        sys.stdout = output_file
        execute_threading(directory, output_file, num_threads=16)


if __name__ == "__main__":
    filename = "data/wuf20-71-M/wuf20-01.mwcnf"
    # filename = "data/wuf100-430-Q/wuf100-01.mwcnf"
    # filename = "data/wuf50-218R-Q/wuf50-013.mwcnf"
    # execute_one_file(filename)

    # with PyCallGraph(output=GraphvizOutput()):
    #     execute_one_file(filename)

    # folder_name = "data/wuf20-71-M"
    # execute_threading(folder_name, num_threads=8)

    run_all()

