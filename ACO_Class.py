import numpy as np  # Major use: Arrays and Matrices\
import math
import time


class AntColonyOptimization:

    def __init__(self, Filepath, threshold_time, inc_ant_count_factor=0.5, rho=0.2, k=1, MAX_TIME=2000, seed = 0):

        self.Filepath = Filepath
        self.FileDict = {}
        self.Dimension = 0
        self.Nodes = ()
        self.EdgeWeights = []
        self.points = []
        self.threshold_time = threshold_time
        self.min_val = math.inf
        self.trace_file = []
        self.alpha = 1
        self.beta = 1
        self.inc_ant_count_factor = inc_ant_count_factor
        self.rho = rho
        self.k = k
        self.MAX_TIME = MAX_TIME
        self.visited_cities = {}
        self.solution = {}
        self.rng = np.random.default_rng(seed)

        # calling functions to populate necessary members
        self.TspToDict()
        self.FindEdgeWeights()
        self.set_alpha_beta()
        self.q = np.mean(self.EdgeWeights)
        self.ants = self.Dimension
        self.cost_matrix = np.zeros(self.ants)
        self.tau = 1 / (self.ants * self.Dimension)
        self.tau_array = self.generate_tau()
        self.delta_tau_array = np.zeros((self.ants, self.Dimension, self.Dimension))
        self.prob_array = np.zeros((self.ants, self.Dimension, self.Dimension))

    # populate all the attributes of the object
    def TspToDict(self):

        File = open(self.Filepath, 'r')
        Name = File.readline().strip().split()[1]
        self.FileDict["Name"] = Name
        Comment = File.readline().strip().split()[1]
        self.FileDict["Comment"] = Comment
        Dimension = int(File.readline().strip().split()[1])
        self.FileDict["Dimension"] = Dimension
        self.Dimension = Dimension
        EdgeType = File.readline().strip().split()[1]
        self.FileDict["EdgeType"] = EdgeType
        File.readline()
        self.Nodes = [[0, 0] for k in range(self.Dimension)]

        for i in range(Dimension):
            NodeList = File.readline().strip().split()
            NodeID = int(NodeList[0]) - 1
            NodeX = float(NodeList[1])
            NodeY = float(NodeList[2])
            self.Nodes[NodeID] = [NodeX, NodeY]
            self.points.append(NodeID)

        self.FileDict["Nodes"] = self.Nodes

    # Find distance between every pair of points
    def FindEdgeWeights(self):
        self.EdgeWeights = [[0 for j in range(self.Dimension)] for i in range(self.Dimension)]
        for i in range(self.Dimension):
            X1 = self.Nodes[i][0]
            Y1 = self.Nodes[i][1]
            for j in range(i + 1, self.Dimension):
                X2 = self.Nodes[j][0]
                Y2 = self.Nodes[j][1]
                self.EdgeWeights[i][j] = round(math.sqrt(((X1 - X2) ** 2 + (Y1 - Y2) ** 2)))
                self.EdgeWeights[j][i] = round(math.sqrt(((X1 - X2) ** 2 + (Y1 - Y2) ** 2)))

    def set_alpha_beta(self):
        """ Returns values of alpha and beta """

        # Setting Alpha and Beta values
        self.alpha = 1
        self.beta = 1
        if self.Dimension < 50:
            self.alpha = 1.11
            self.beta = 1.44
        elif 50 <= self.Dimension < 100:
            self.alpha = 0.954
            self.beta = 1.154
        elif 100 <= self.Dimension < 150:
            self.alpha = 1.04
            self.beta = 0.925
        elif 150 <= self.Dimension:
            self.alpha = 0.75
            self.beta = 1.175

    def prob_matrix(self, specific_ant):
        """ Returns a matrix that has values of probability of an ant travelling between two cities """

        # We alter the distance and pheromone array such that values corresponding to cities already traveled are 0
        # The formula for probability of traveling between city i and j is given by:
        #               (Tij(k)^a)/(dij^b)
        # Pij(k) = ----------------------------------
        #            SUM_j((Tij(k)^a)/(dij^b))
        # Where T: tau, d: distance or cost, a: alpha, b: beta, k: k-th ant(iteration) of the algorithm

        no_city = int(self.Dimension)
        prob_array = np.zeros((no_city, no_city))
        for r in range(no_city):
            for c in range(no_city):
                if (r != c) and (c + 1 not in self.visited_cities[specific_ant]):
                    prob_array[r][c] = (self.tau_array[r][c] ** self.alpha) / (self.EdgeWeights[r][c] ** self.beta)
        for r in range(no_city):
            if np.sum(prob_array[r]) != 0:
                prob_array[r] /= np.sum(prob_array[r])

        return prob_array

    def city_selection(self, specific_ant, current_city):
        """ Returns the next city number based on maximum probability """

        random_value = self.rng.uniform(0, 1)
        sum_prob = 0
        for p in range(len(self.prob_array[specific_ant][current_city - 1])):
            sum_prob += self.prob_array[specific_ant][current_city - 1][p]
            if random_value < sum_prob:
                next_city = p + 1
                # return next_city
                break
        if sum_prob == 0:
            return 'skip'
        else:
            return next_city

    def total_cost(self, specific_ant):
        """ Calculates the total cost of travel, based on total distance travelled """

        total_dist = 0
        for c in range(len(self.visited_cities[specific_ant]) - 1):
            curr_city = self.visited_cities[specific_ant][c]
            curr_city_index = curr_city - 1
            next_city = self.visited_cities[specific_ant][c + 1]
            next_city_index = next_city - 1
            total_dist += self.EdgeWeights[curr_city_index][next_city_index]

        return total_dist

    def pheromone_update(self):
        """ Updates the pheromone for the arcs between the cities """

        up_tau = np.zeros((self.Dimension, self.Dimension))
        for i in range(self.Dimension):
            for j in range(self.Dimension):
                sum_del_tau = 0
                for a in range(self.ants):
                    sum_del_tau += self.delta_tau_array[a][i][j]
                up_tau[i][j] = (1 - self.rho) * self.tau_array[i][j] + sum_del_tau
                # print(up_tau[i][j], (1 - rho), sum_del_tau, (1 - rho) * up_tau[i][j] + sum_del_tau, tau_array[i][j])

        # print(up_tau)
        return up_tau

    def generate_tau(self):
        """ Initializes the Tau matrix to the set tau value """

        self.tau_array = np.full((self.Dimension, self.Dimension), self.tau)
        # print(tau_array)
        np.fill_diagonal(self.tau_array, 0)

        return self.tau_array

    def RUN_ACO(self):
        start_time = time.time()
        # print('Program Started at: ', start_time)
        count_stuck_solution = 0
        max_solution_count = 3
        start_t = 0
        # prev_best_solution = math.inf
        for t in range(self.MAX_TIME):

            if t > 0:
                prev_best_solution = self.min_val

            # Zeroth time step
            Cities = np.linspace(1, self.Dimension, self.Dimension, dtype=int)
            for a in range(self.ants):
                start_city = self.rng.choice(Cities)  # Placing the ant on a random city
                self.visited_cities[a] = [start_city]  # Adding the city to the visited cities
            # print(self.visited_cities)

            # Time step 1 to MAX_TIME
            while len(self.visited_cities[self.ants - 1]) < self.Dimension:
                for a in range(self.ants):

                    if len(self.visited_cities[a]) < self.Dimension:
                        current_city = self.visited_cities[a][-1]
                        current_city_index = current_city - 1

                        # Prob Calculated
                        self.prob_array[a] = self.prob_matrix(a)

                        # City selected & Updated
                        next_city = self.city_selection(a, current_city)
                        # print(next_city)
                        if next_city == 'skip':
                            continue
                        self.visited_cities[a].append(next_city)
                        next_city_index = next_city - 1

                        self.delta_tau_array[a][current_city_index][next_city_index] = self.q / self.total_cost(a)
                # print(self.visited_cities)
                self.tau_array = self.pheromone_update()

            self.cost_matrix = np.zeros(self.ants)
            for a in range(self.ants):
                self.visited_cities[a].append(self.visited_cities[a][0])
                self.cost_matrix[a] = self.total_cost(a)

            min_cost = math.inf
            solution_ant = -10
            for a in range(self.ants):
                if self.cost_matrix[a] < min_cost:
                    min_cost = self.cost_matrix[a]
                    solution_ant = a
            self.visited_cities[solution_ant] = np.array(self.visited_cities[solution_ant])-1
            self.solution[min_cost] = list(self.visited_cities[solution_ant])
            # print(self.solution[min_cost])
            # solution[min(cost_matrix)] = visited_cities[int(np.where(cost_matrix == min(cost_matrix)))]

            iter_time = time.time()
            time_elapsed = iter_time - start_time
            # print('Time Elapsed: ', time_elapsed)

            for val, cities in self.solution.items():
                if val < self.min_val:
                    self.min_val = val
                    # starting_t = t

            self.trace_file.append([round(time_elapsed, 2), self.min_val])
            # print('Best Solution found so far: ')
            # print(self.min_val)
            # print(self.solution[self.min_val])

            if t > 0:
                if prev_best_solution == self.min_val:
                    count_stuck_solution += 1
                    if count_stuck_solution > max_solution_count:
                        self.ants += int(self.inc_ant_count_factor * self.ants)
                        # The array contains Pheromone values changes of arcs between the cities for each ant
                        self.delta_tau_array = np.append(self.delta_tau_array,
                                                    np.zeros((int(self.inc_ant_count_factor * self.ants), self.Dimension, self.Dimension)),
                                                    axis=0)
                        self.prob_array = np.append(self.prob_array,
                                               np.zeros((int(self.inc_ant_count_factor * self.ants), self.Dimension, self.Dimension)),
                                               axis=0)
                        # alpha = ra.uniform(0, 2)
                        # beta = ra.uniform(0, 2)
                        count_stuck_solution = 0
                        max_solution_count = t - start_t + 3
                        start_t = t

            if iter_time - start_time > self.threshold_time:
                break
