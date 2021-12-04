## Import packages
import os
import time
import math
import pandas
import argparse
from PROGRAM import BranchAndBound
from APPROXIMATION import NearestNeighbour
from ACO_Class import AntColonyOptimization
from SA import SimulatedAnnealing

def SaveWeightAndPath(args, MinWeight, BestPath, deterministic=True):
    '''
    Inputs: 
        MinWeight: Int -->  Minimum weight of the path found by the algorithm
        BestWeight: List --> Order in which the nodes should be traversed
    Output:
        Saved file with the appropriate name
    '''
    if deterministic: OutPath = args.inst + "_" + args.alg + "_" + args.time + ".sol"
    else: OutPath = args.inst + "_" + args.alg + "_" + args.time + "_" + args.seed + ".sol"
    with open(OutPath, 'w') as OutFile:
        OutFile.write(str(MinWeight) + '\n')
        for i in range(len(BestPath)):
            Node = BestPath[i]
            if i != len(BestPath) - 1: OutFile.write(str(Node) + ',')
            else: OutFile.write(str(Node) + '\n')
    return


def SaveTraces(args, BestTraces, deterministic=True):
    '''
    Inputs:
        BestTraces: List --> Traces containing [time, quality] found by the algorithm
    Output:
        Saved file with the appropriate name
    '''
    if deterministic: OutPath = args.inst + "_" + args.alg + "_" + args.time + ".trace"
    else: OutPath = args.inst + "_" + args.alg + "_" + args.time + "_" + args.seed + ".trace"
    with open(OutPath, 'w') as OutFile:
        for i in range(len(BestTraces)):
            Time = BestTraces[i][0]; OutFile.write(str(Time) + ',')
            Quality = BestTraces[i][1]; OutFile.write(str(Quality) + '\n')
            # Error = BestTraces[i][2]; OutFile.write(str(Error) + '\n')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', required=True, help='filename')
    parser.add_argument('-alg', required=True, help='BnB | Approx | LS1 | LS2')
    parser.add_argument('-time', required=True, help='cutoff_in_seconds')
    parser.add_argument('-seed', default="0", help='random_seed')
    args = parser.parse_args()

    Filepath = "./DATA/" + args.inst + ".tsp"
    Cutoff = int(args.time)
    Dataframe = pandas.read_csv("./DATA/solutions.csv", sep=",")
    print(Dataframe.head())
    index = Dataframe.index[Dataframe['Instance'] == args.inst][0]
    OptWeight = Dataframe.iloc[index, 1]

    ## Branch and bound algorithm
    if args.alg == 'BnB':
        BnB = BranchAndBound(Filepath, Cutoff)
        BnB.Main(OptWeight)
        print(BnB.MinWeights[-1])
        print(BnB.BestPaths[-1])
        print(BnB.BestTraces[-1])
        # Save min weight and best path 
        SaveWeightAndPath(args, BnB.MinWeights[-1], BnB.BestPaths[-1])         
        # Save best traces
        SaveTraces(args, BnB.BestTraces)

    if args.alg == 'Approx':
        Approx = NearestNeighbour(Filepath, Cutoff)
        Approx.find_approximate_solution()
        print(Approx.best_sol)
        print(Approx.best_path)
        print(Approx.trace)
        # Save min weight and best path
        SaveWeightAndPath(args, Approx.best_sol, Approx.best_path)
        # Save best traces
        SaveTraces(args, Approx.trace)

    if args.alg == 'LS1':
        ACO = AntColonyOptimization(Filepath, Cutoff, seed = int(args.seed))
        ACO.RUN_ACO()
        print(ACO.min_val)
        print(ACO.solution[ACO.min_val])
        print(ACO.trace_file)
        # Save min weight and best path
        SaveWeightAndPath(args, ACO.min_val, ACO.solution[ACO.min_val], deterministic=False)
        # Save best traces
        SaveTraces(args, ACO.trace_file, deterministic=False)

    if args.alg == 'LS2':
        SA = SimulatedAnnealing(Filepath, Cutoff, seed = int(args.seed))
        SA.run_simulation()
        print(SA.best_distance)
        print(SA.best_solution)
        print(SA.trace)
        # Save min weight and best path
        SaveWeightAndPath(args, SA.best_distance, SA.best_solution, deterministic=False)
        # Save best traces
        SaveTraces(args, SA.trace, deterministic=False)