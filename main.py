import os
import random
from solve import Formula


def create_random_config(number_of_variables):
    config = []
    for _ in range(number_of_variables):
        if random.random() < 0.5:
            config.append(1)
        else:
            config.append(0)
    return config


def main():
    filename = os.path.join(os.getcwd(), "data", "wuf20-71-M", "wuf20-01.mwcnf")
    formula = Formula(filename)
    variable_configuration = create_random_config(formula.number_of_variables)
    print(formula)
    print("Assignment: ", variable_configuration)
    results = formula.get_clauses_results(variable_configuration)
    print("Clauses results:",results)
    success_percentage = formula.get_success_rate(variable_configuration) * 100
    # print it with two decimal places
    print(f"Number of true clauses: { round(success_percentage, 2) } %")

if __name__ == "__main__":
    main()