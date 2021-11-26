import math
import aco_lib as aco
import numpy as np
import random as ra
import time

# Ant Colony Optimization in Traveling Salesman Problem
# Assuming that the salesman has to start from a given city (input)
# and has to return to the same city at the end of the trip ..
# and all the costs are symmetric,
# i.e, cost for traveling from city i to j is same as that of traveling from city j to i ..
# each city can be visited only once

threshold_time = 150000

Filepath = "./DATA/Cincinnati.tsp"
FileDict = aco.TspToDict(Filepath)
Dimension = FileDict["Dimension"]
Nodes = FileDict["Nodes"]
EdgeWeights = np.array(aco.FindEdgeWeights(Dimension, Nodes))

ants = Dimension # Number of ants (iterations), this becomes the termination criteria
tau = 1/(ants*Dimension)  # Initial pheromone value of arc between cities
inc_ant_count_factor = 0.5
rho = 0.2  # Evaporation coefficient (if taken 0, its for simplicity)
alpha, beta = aco.set_alpha_beta(Dimension)  # Alpha and Beta value, used in probability calculation
# alpha = 1
# beta = 1  # Beta value, used in probability calculation
q = np.mean(EdgeWeights)  # Pheromone count (used while updating the pheromone of the arcs after an iteration)
k = 1  # Distance to cost conversion parameter C = k*D (Assumed linear behaviour)
MAX_TIME = 2000 # Time steps for ants to travel from one city to other


# Main program starts

delta_tau_array = np.zeros((ants, Dimension, Dimension))  # The array contains Pheromone values changes
# of arcs between the cities for each ant
tau_array = aco.generate_tau(Dimension, tau)
prob_array = np.zeros((ants, Dimension, Dimension))
visited_cities = {} # To memorize the cities visited by a particular ant
solution = {}
trace_file = []

start_time = time.time()
# print('Program Started at: ', start_time)
count_stuck_solution = 0
max_solution_count = 3
start_t = 0
# prev_best_solution = math.inf
for t in range(MAX_TIME):

    if t > 0:
        prev_best_solution = min_val

    # Zeroth time step
    Cities = np.linspace(1, Dimension, Dimension, dtype=int)
    for a in range(ants):
        start_city = ra.choice(Cities)  # Placing the ant on a random city
        visited_cities[a] = [start_city]  # Adding the city to the visited cities

    # Time step 1 to MAX_TIME
    while len(visited_cities[ants-1]) < Dimension:
        for a in range(ants):

            if len(visited_cities[a]) < Dimension:
                current_city = visited_cities[a][-1]
                current_city_index = current_city-1

                # Prob Calculated
                prob_array[a] = aco.prob_matrix(Dimension, EdgeWeights, tau_array, alpha, beta, visited_cities, a)

                # City selected & Updated
                next_city = aco.city_selection(prob_array, a, current_city)
                # print(next_city)
                if next_city == 'skip':
                    continue
                visited_cities[a].append(next_city)
                next_city_index = next_city-1

                delta_tau_array[a][current_city_index][next_city_index] = q/aco.total_cost(visited_cities, a, EdgeWeights)
        # print(visited_cities[a])
        tau_array = aco.pheromone_update(delta_tau_array, ants, tau_array, Dimension, rho)

    cost_matrix = np.zeros(ants)
    for a in range(ants):
        visited_cities[a].append(visited_cities[a][0])
        cost_matrix[a] = aco.total_cost(visited_cities, a, EdgeWeights)

    min_cost = math.inf
    solution_ant = -10
    for a in range(ants):
        if cost_matrix[a] < min_cost:
            min_cost = cost_matrix[a]
            solution_ant = a
    solution[min_cost] = visited_cities[solution_ant]
    # solution[min(cost_matrix)] = visited_cities[int(np.where(cost_matrix == min(cost_matrix)))]

    iter_time = time.time()
    time_elapsed = iter_time-start_time
    print('Time Elapsed: ', time_elapsed, 'Ants: ', ants, 'Iteration: ', t)

    min_val = math.inf
    for val, cities in solution.items():
        if val < min_val:
            min_val = val
            # starting_t = t

    trace_file.append([min_val, round(time_elapsed, 2)])
    print('Best Solution found so far: ')
    print(min_val)
    print(solution[min_val])

    if t > 0:
        if prev_best_solution == min_val:
            count_stuck_solution += 1
            if count_stuck_solution > max_solution_count:
                ants += int(inc_ant_count_factor*ants)
                # The array contains Pheromone values changes of arcs between the cities for each ant
                delta_tau_array = np.append(delta_tau_array, np.zeros((int(inc_ant_count_factor*ants), Dimension, Dimension)), axis=0)
                prob_array = np.append(prob_array, np.zeros((int(inc_ant_count_factor*ants), Dimension, Dimension)), axis=0)
                # alpha = ra.uniform(0, 2)
                # beta = ra.uniform(0, 2)
                count_stuck_solution = 0
                max_solution_count = t - start_t + 3
                start_t = t

    if iter_time-start_time > threshold_time:
        break
