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

    def is_clause_true(self, clause: list[int], configuration: list[int]) -> bool:
        # clause is in CNF form
        # we need to calculate boolean value
        # make this faster
        for literal in clause:
            if literal < 0:
                literal_index = abs(literal) - 1
                if not configuration[literal_index]:
                    return True
            else:
                literal_index = literal - 1
                if configuration[literal_index]:
                    return True
        return False


    def get_clauses_results(self, configuration: list[int]) -> list[bool]:
        # use is_clause_true to calculate all clauses
        # computationaly expensive operation O(n*m)
        # where n is number of clauses and m is number of literals in clause
        return [self.is_clause_true(clause, configuration) for clause in self.clauses]

    def does_satisfy(self, configuration: list[int]) -> bool:
        # find only if the formula is true or false
        # worst case O(n*m) average case O(n)
        # where n is number of clauses and m is number of literals in clause
        for clause in self.clauses:
            if not self.is_clause_true(clause, configuration):
                return False
        return True

    def get_total_weight(self, configuration: list[int]) -> int:
        # calculate total weight as a sum of weights that are true in configuration
        # time complexity O(n)
        # where n is number of literals
        # return sum([self.weights[i] for i, value in enumerate(configuration) if value])

        # make this faster
        total_weight = 0
        for i, value in enumerate(configuration):
            if value:
                total_weight += self.weights[i]
        return total_weight


    def get_success_rate(self, configuration: list[int]) -> float:
        # calculate success rate using self.get_clauses_results
        return sum(self.get_clauses_results(configuration)) / len(self.clauses)

    def get_mapped_weight(self, configuration: list[int]) -> float:
        # map success rate between 0 and 1
        # where 1 is max possible value
        max_value = sum(self.weights)
        return self.get_total_weight(configuration) / max_value
