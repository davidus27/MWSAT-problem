class Formula:
    def __init__(self, from_file):
        self.clauses = []
        self.weights = []
        self.number_of_variables = 0

        self.__load_file(from_file)

    def __str__(self):
        return "Clauses: " + str(self.clauses) + "\nWeights: " + str(self.weights) + "\nNumber_of_variables: " + str(self.number_of_variables)

    def __load_file(self, from_file: str):
        with open(from_file, "r") as f:
            for line in f:
                if line[0] == 'c':
                    continue
                elif line[0] == 'p':
                    self.number_of_variables = int(line.split()[2])
                elif line[0] == 'w':
                    weights = line.split()[1:]
                    self.weights = [int(i) for i in weights[:-1]]
                else:
                    self.clauses.append([int(i) for i in line.split()[:-1]])

    def get_clause_and_weight(self, clause_index: int):
        return self.clauses[clause_index], self.weights[clause_index]

    def get_clause_weight(self, clause: list[int], configuration: list[int]):
        if len(configuration) != self.number_of_variables:
            raise ValueError(
                "Assignment length does not match number of variables")

        weight = 0
        for literal in clause:
            literal_index = abs(literal) - 1
            if configuration[literal_index]:
                weight += self.weights[literal_index]
        return weight

    def get_clause_weight_by_index(self, clause_index: int, configuration: list[int]):
        if len(configuration) != self.number_of_variables:
            raise ValueError(
                "Assignment length does not match number of variables")

        weight = 0
        for literal in self.clauses[clause_index]:
            literal_index = abs(literal) - 1
            if configuration[literal_index]:
                weight += self.weights[literal_index]
        return weight

    def is_clause_true(self, clause: list[int], configuration: list[int]) -> bool:
        # clause is in CNF form
        # we need to calculate boolean value
        final_value = False
        for literal in clause:
            if literal < 0:
                literal_index = abs(literal) - 1
                literal_value = not configuration[literal_index]
            else:
                literal_index = literal - 1
                literal_value = bool(configuration[literal_index])

            final_value = final_value or literal_value

            if final_value:
                break

        return final_value

    def get_clauses_results(self, configuration: list[int]) -> list[bool]:
        # use is_clause_true to calculate all clauses
        return [self.is_clause_true(clause, configuration) for clause in self.clauses]

    def get_total_weight(self, configuration: list[int]) -> int:
        # calculate total weight using self.get_clauses_results
        return sum([self.get_clause_weight(clause, configuration) * self.is_clause_true(clause, configuration) for clause in self.clauses])

    def get_success_rate(self, configuration: list[int]) -> float:
        # calculate success rate using self.get_clauses_results
        return sum(self.get_clauses_results(configuration)) / len(self.clauses)

    def get_mapped_weight(self, configuration: list[int]) -> float:
        # map success rate between 0 and 1
        # where 1 is max possible value
        
        max_value = self.get_total_weight([1] * self.number_of_variables)
        return self.get_total_weight(configuration) / max_value

    def get_punished_weight(self, configuration: list[int], punishment_coefficient=1):
        sum_value = 0
        results = self.get_clauses_results(configuration)
        for clause_index, clause_result in enumerate(results):
            if clause_result == 1:
                sum_value += self.get_clause_weight_by_index(
                    clause_index, configuration)
            else:
                sum_value -= self.get_clause_weight_by_index(
                    clause_index, configuration) * punishment_coefficient

        return sum_value