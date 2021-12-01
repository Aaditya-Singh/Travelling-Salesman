import math
import time
from collections import defaultdict
import numpy as np


class NearestNeighbour:
    """
    Class for the Nearest Neighbour Approximation Algorithm
    """
    def __init__(self, Filepath, Cutoff):
        self.Filepath = Filepath
        self.locations, self.meta, self.n_locations = self.populate_locations()
        self.visited = np.zeros(self.n_locations)
        self.path = []
        self.quality = 0
        self.best_sol = float('inf')
        self.best_path = None
        self.trace = []
        self.cutoff = Cutoff

    def populate_locations(self):
        """
        Populates the locations list in which each element is tuple: (location_index, x_coordinate, y_coordinate)
        :return: meta_information of the instance, populated location list, no. of locations
        """
        locations, meta = [], defaultdict()
        with open(self.Filepath, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "NODE_COORD_SECTION": break
                key, value = line.split(": ")
                meta[key] = value
            n = int(meta["DIMENSION"])
            for i in range(n):
                index, x, y = fp.readline().split()
                locations.append((int(index), float(x), float(y)))
        return locations, meta, n

    def reset(self):
        """
        Reset the visited, path arrays and the quality variable
        :return:
        """
        self.visited = np.zeros(self.n_locations)
        self.path = []
        self.quality = 0

    def find_approximate_solution(self):
        """
        Finds the approximate solution by calling the helper_find_approximate_solution routine for each location
        """
        start_time = time.time()
        for i in range(self.n_locations):
            self.helper_find_approximate_solution(i + 1)
            if self.quality < self.best_sol:
                self.best_path = self.path
                self.best_sol = self.quality

                end_time = time.time()
                self.trace.append([end_time - start_time, self.quality])
                if end_time - start_time > self.cutoff:
                    break

            self.reset()

    def helper_find_approximate_solution(self, origin):
        """
        Finds the best neighbouring solution when starting from the "origin" location
        :param origin: location index to begin with
        """
        self.visited[origin - 1] = 1
        curr = origin
        self.path.append(origin)

        while np.count_nonzero(self.visited) < self.n_locations:
            min_dist, l = self.find_closest_location(curr)
            self.visited[l - 1] = 1
            self.path.append(l)
            self.quality += min_dist
            curr = l

        curr_x, curr_y = self.locations[curr - 1][1], self.locations[curr - 1][2]
        orig_x, orig_y = self.locations[origin - 1][1], self.locations[origin - 1][2]

        self.quality += math.sqrt((curr_x - orig_x) ** 2 + (curr_y - orig_y) ** 2)  # circling back to the starting
        # location

    def find_closest_location(self, curr_location):
        """
        Finds the closest location to the "curr_location" by using L2 distance

        :param curr_location: location index of the concerned location
        :return: the minimum distance and the corresponding location
        """
        min_distance = float('inf')
        curr_location_index, curr_x, curr_y = self.locations[curr_location - 1]
        closest_location = None

        for location_index, x, y in self.locations:
            if not self.visited[location_index - 1]:
                dist = math.sqrt((curr_x - x) ** 2 + (curr_y - y) ** 2)
                if dist < min_distance:
                    min_distance = dist
                    closest_location = location_index

        return min_distance, closest_location
