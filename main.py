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
    return GeneticAlgorithm(population_size=500, reproduction_count=250, new_blood=80, elite_size=30, survivors=5, max_iterations=350) \
        .set_initial_population_method(RandomInitialPopulation()) \
        .set_fitness_function(LogisticFitnessFunction(strict_satisfiability=False)) \
        .set_selection_method(TournamentSelection()) \
        .set_crossover_method(SinglePointCrossover()) \
        .set_mutation_method(BitFlipMutation(mutation_chance=0.01))


def solve_for_file(filename: str, evolution_algorithm: GeneticAlgorithm):
    formula = Formula(filename)

    solution = evolution_algorithm.solve(formula)
    success_rate = formula.get_success_rate(solution)
    total_weight = formula.get_total_weight(solution)
    perfomance = evolution_algorithm.get_perfomance_tracker()
    
    # print("Solving file:", filename)
    # print("Solution:", solution)
    # print("Success rate:", success_rate)
    # print("Weight:", total_weight)

    return {
        "solution": solution,
        "success_rate": success_rate,
        "total_weight": total_weight,
        "perfomances": perfomance,
        "success": success_rate == 1.0,
    }

def execute_one_file(filename: str, output_file):
    formula = Formula(filename)
    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()
    print("Solving file:", filename)
    # send output to file
    if output_file != None:
        sys.stdout = output_file

    solution = evolution_algorithm.solve(formula)

    print("Solution:", solution)
    print("Success rate:", formula.get_success_rate(solution))
    print("Weight:", formula.get_total_weight(solution))

def thread_function(folders: list[str]):
    # create evolution algorithm
    evolution_algorithm = buildEvolutionAlgorithm()
    max_retries = 2

    for directory in folders:
        basename = os.path.basename(directory)
        output_file = open(os.path.join("results", basename), "w")

        files = os.listdir(directory)
        files = [os.path.join(directory, file) for file in files]
        
        # solve for 100 files
        for filename in files[:100]:
            for _ in range(max_retries):
                solution = solve_for_file(filename, evolution_algorithm)
                output_file.write(str(solution))
                if solution["success"]:
                    break
        output_file.close()

def execute_threading(folder_name: str, num_threads=4):
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

def run_all(num_threads=4):
    # run threaded for all directories in data

    # get all directories in data that starts with wuf
    data_dir = "data"
    directories = os.listdir(data_dir)
    directories = [directory for directory in directories if directory.startswith("wuf")]
    # removes directories that contains R-
    directories = [directory for directory in directories if "-R" in directory]
    directories = [os.path.join(data_dir, directory) for directory in directories]
    directories = [directory for directory in directories if os.path.isdir(directory)]
    
    # create a pool of threads
    # we will have num_threads of threads and n directories
    # each thread will get n/num_threads directories
    threads = []
    for i in range(num_threads):
        x = threading.Thread(target=thread_function, args=(directories[i::num_threads],))
        threads.append(x)
        print("Starting thread", i)
        x.start()


if __name__ == "__main__":
    # with PyCallGraph(output=GraphvizOutput()):
    #     execute_one_file(filename)

    # folder_name = "data/wuf20-71-M"
    # execute_threading(folder_name, num_threads=2)

    run_all()

