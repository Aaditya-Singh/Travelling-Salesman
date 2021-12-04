# Travelling Salesman (CSE 6140 Fall 2021 Project)


## Group members
- Mayank Lunayach 
- Kshitij Pisal
- Aaditya Singh
- Vasistha Vinod


## Description
Solve Travelling Salesman Problem with the following algorithms:  
1. Exact: Branch and Bound
2. Approximation: Nearest-Neighbor
3. Local Search 1: Ant Colony Optimization
4. Local Search 2: Simulated Annealing


## Usage:
```
python tsp_main.py -inst <filename>
                    -alg [BnB | Approx | LS1 | LS2]
                    -time <cutoff_in_seconds>
                    [-seed <random_seed>]
```  
- alg: the method to use, BnB is branch and bound, Approx is approximation, LS1 is local search 1, and LS2 is local search 2.
- inst: the filepath of a single input instance.
- time: the cut-off time in seconds.
- seed: the random seed used to seed any random generator used for solving TSP.


## Dependencies
- Python  3.7.11
- NumPy 1.17.4
- Pandas  1.3.4  


### Branch and Bound
    Implemented in PROGRAM.py and linked to tsp_main.py.
    Create BranchAndBound class instance and call BranchAndBound.Main() to run the algorithm.
    BranchAndBound.MinWeights[-1] returns the weight of the best path found.
    BranchAndBound.BestPaths[-1] returns the best path found.
    BranchAndBound.BestTraces[-1] returns the time and quality of subsequently better solutions.
    
    
### Approximation
    Implemented in APPROXIMATION.py and linked to tsp_main.py.
    Create NearestNeighbour class instance and call NearestNeighbour.find_approximate_solution() to run the algorithm.
    NearestNeighbour.best_sol returns the weight of the best path found.
    NearestNeighbour.best_path returns the best path found.
    NearestNeighbour.trace returns the time and quality of subsequently better solutions.
    
    
### Local Search 1
    Implemented in ACO_Class.py and linked to tsp_main.py.
    Create AntColonyOptimization class instance and call AntColonyOptimization.RUN_ACO() to run the algorithm.
    AntColonyOptimization.min_val returns the weight of the best path found.
    AntColonyOptimization.solution[AntColonyOptimization.min_val] returns the best path found.
    AntColonyOptimization.trace_file returns the time and quality of subsequently better solutions.
    
    
### Local Search 2
    Create SimulatedAnnealing class instance and call SimulatedAnnealing.run_simulation() to run the algorithm.
    SimulatedAnnealing.best_distance returns the weight of the best path found.
    SimulatedAnnealing.best_solution returns the best path found.
    SimulatedAnnealing.trace returns the time and quality of subsequently better solutions.