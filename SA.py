import numpy as np
import math, time
import os

class SimulatedAnnealing():
    
    # Class constructor with default values for the cooling rate, initial temperature and random generator's seed
    def __init__(self, Filepath, cutoff, cool_rate=1e-7, T0 = 10000, seed = 10):
        
        self.Filepath = Filepath
        self.FileDict = {}
        self.Dimension = 0
        self.Nodes = list()
        self.EdgeWeights = list()
        self.cutoff = cutoff
        self.points = list()
        self.best_solution = None
        self.best_distance = np.inf
        self.trace = list()
        self.Temp = T0
        self.cool_rate = cool_rate
        
        # instantiating the random number generator object with the given seed
        self.rng = np.random.default_rng(seed)
        
        # calling functions to populate necessary members
        self.TspToDict()
        self.FindEdgeWeights()

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
            for j in range(i+1, self.Dimension):
                X2 = self.Nodes[j][0] 
                Y2 = self.Nodes[j][1]
                self.EdgeWeights[i][j] = round(math.sqrt(((X1 - X2)**2 + (Y1 - Y2)**2)))
                self.EdgeWeights[j][i] = round(math.sqrt(((X1 - X2)**2 + (Y1 - Y2)**2)))

    # calculate total distance of a given route
    def route_distance(self, route):
        
        n = len(route)
        dist = 0
        for i in range(n-1):
            dist += self.EdgeWeights[int(route[i])][int(route[i+1])]

        # complete the loop of the route
        dist += self.EdgeWeights[int(route[n-1])][int(route[0])]
        return dist

    # swapping 2 randomly chosen points to generate neighboring solution
    def get_nbr_soln(self, route, double_swap=False):
        
        new_route = np.copy(route)
        
        if double_swap:
            # choosing 4 random points
            [a, b, x, y] = self.rng.choice(range(len(new_route)), 4, replace = False)
            #print(a, b, x, y)
        
            # swapping the two pairs of points in the given path
            new_route[[a, b]] = new_route[[b, a]]
            new_route[[x, y]] = new_route[[y, x]]
        
        else:
            
            # choosing 2 random points
            [x, y] = self.rng.choice(range(len(new_route)), 2, replace = False)
            #print(x, y)
            
            # swapping the two points in the given path
            new_route[[x, y]] = new_route[[y, x]]
        
        return new_route

    # generating the probability of accepting neighboring solution
    def acceptance_criterion(self, curr_dist, new_dist):
        if curr_dist > new_dist:
            return 1.0
        else:
            return np.exp((curr_dist - new_dist)/self.Temp)

    # Function that runs the simulation
    def run_simulation(self):
        
        # store time of start of the SA algorithm
        start_time = time.time()

        # Randomly initiate the current solution and best solution
        current_solution = np.copy(self.points)
        self.rng.shuffle(current_solution)
        self.best_solution = np.copy(current_solution)
        
        # to keep track of number of iterations since last update of best solution
        numiter_since_new_best = 0

        while self.Temp > 0:
            
            # indicator variable for using double swap to find neighboring solution
            double_swap = False
            
            # if the best solution has not been updated for 500 iterations,
            # we use double swap to get a neighboring solution
            if numiter_since_new_best >= 500:
                double_swap = True
                
            # neighboring solution
            new_solution = self.get_nbr_soln(current_solution, double_swap)

            # distance of new route
            current_dist = self.route_distance(current_solution)
            new_dist = self.route_distance(new_solution)
            
            # incrementing the iteration counter
            numiter_since_new_best += 1

            # Deciding whether or not to move to neighboring solution
            p = self.acceptance_criterion(current_dist, new_dist)
            if p > self.rng.uniform(0, 1):
                current_solution = new_solution
                # updating best solution if current is better
                if self.route_distance(current_solution) < self.route_distance(self.best_solution):
                    self.best_solution = current_solution
                    self.best_distance = int(self.route_distance(self.best_solution))
                    
                    # resetting the number of iterations since last update of best solution
                    numiter_since_new_best = 0
                    
                    # time elapsed since start of the algorithm
                    elapsed = time.time() - start_time
                    
                    # adding the new best distance found to the trace
                    self.trace.append([round(elapsed,2), self.best_distance])
                
            # cooling down
            self.Temp *= (1-self.cool_rate)

            # time elapsed since start of the algorithm
            elapsed = time.time() - start_time
            if elapsed > self.cutoff:
                #print("Program was cutoff")
                return 