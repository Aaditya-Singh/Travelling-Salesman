import numpy as np  # Major use: Arrays and Matrices
import random  # Major use: Random number generation


def prob_matrix(no_city, dist_array, tau_array, alpha, beta, city_trav, specific_ant):
    """ Returns a matrix that has values of probability of an ant travelling between two cities """

    # We alter the distance and pheromone array such that values corresponding to cities already traveled are 0
    # The formula for probability of traveling between city i and j is given by:
    #               (Tij(k)^a)/(dij^b)
    # Pij(k) = ----------------------------------
    #            SUM_j((Tij(k)^a)/(dij^b))
    # Where T: tau, d: distance or cost, a: alpha, b: beta, k: k-th ant(iteration) of the algorithm

    prob_array = np.zeros((no_city, no_city))
    for r in range(no_city):
        for c in range(no_city):
            if r != c:
                prob_array[r][c] = (tau_array[r][c] ** alpha) / (dist_array[r][c] ** beta)
    curr_city_index = city_trav[specific_ant][-1] - 1
    for c in city_trav[specific_ant]:
        prob_array[curr_city_index][c - 1] = 0
    for r in range(no_city):
        if np.sum(prob_array[r]) != 0:
            prob_array[r] /= np.sum(prob_array[r])

    return prob_array


def city_selection(prob_array, specific_ant, current_city):
    """ Returns the next city number based on maximum probability """

    random_value = random.uniform(0, 1)
    sum_prob = 0
    for p in range(len(prob_array[specific_ant][current_city - 1])):
        sum_prob += prob_array[specific_ant][current_city - 1][p]
        if random_value < sum_prob:
            next_city = p + 1
            break

    # Current city index = current_city - 1
    # return int(np.where(prob_array[specific_ant][current_city-1] == np.amax(prob_array[specific_ant][current_city-1]))[0])+1
    return next_city


def total_cost(city_trav, specific_ant, dist_array):
    """ Calculates the total cost of travel, based on total distance travelled """

    total_dist = 0
    for c in range(len(city_trav[specific_ant]) - 1):
        curr_city = city_trav[specific_ant][c]
        curr_city_index = curr_city - 1
        next_city = city_trav[specific_ant][c + 1]
        next_city_index = next_city - 1
        total_dist += dist_array[curr_city_index][next_city_index]

    return total_dist


def pheromone_update(delta_tau_array, ants, tau_array, no_cities, rho):
    """ Updates the pheromone for the arcs between the cities """

    up_tau = np.zeros((no_cities, no_cities))
    for i in range(no_cities):
        for j in range(no_cities):
            sum_del_tau = 0
            for a in range(ants):
                sum_del_tau += delta_tau_array[a][i][j]
            # print(up_tau[i][j], (1-rho), sum_del_tau, (1-rho)*up_tau[i][j] + sum_del_tau)
            up_tau[i][j] = (1 - rho) * tau_array[i][j] + sum_del_tau

    # print(up_tau)
    return up_tau


def generate_tau(Dimension, tau):
    """ Initializes the Tau matrix to the set tau value """

    tau_array = np.full((Dimension, Dimension), tau)
    np.fill_diagonal(tau_array, 0)

    return tau_array