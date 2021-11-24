## Import packages
import os
import time
import math
import pandas
import argparse
from PROGRAM import BranchAndBound


def SaveWeightAndPath(args, MinWeight, BestPath):
    '''
    Inputs: 
        MinWeight: Int -->  Minimum weight of the path found by the algorithm
        BestWeight: List --> Order in which the nodes should be traversed
    Output:
        Saved file with the appropriate name
    '''
    OutPath = args.inst + "_" + args.alg + "_" + args.time + ".sol"
    with open(OutPath, 'w') as OutFile:
        OutFile.write(str(MinWeight) + '\n')
        for i in range(len(BestPath)):
            Node = BestPath[i]
            if i != len(BestPath) - 1: OutFile.write(str(Node) + ',')
            else: OutFile.write(str(Node) + '\n')
    return


def SaveTraces(args, BestTraces):
    '''
    Inputs:
        BestTraces: List --> Traces containing [time, quality] found by the algorithm
    Output:
        Saved file with the appropriate name
    '''
    OutPath = args.inst + "_" + args.alg + "_" + args.time + ".trace"
    with open(OutPath, 'w') as OutFile:
        for i in range(len(BestTraces)):
            Time = BestTraces[i][0]; OutFile.write(str(Time) + ',')
            Quality = BestTraces[i][1]; OutFile.write(str(Quality) + '\n')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', required=True, help='filename')
    parser.add_argument('-alg', required=True, help='BnB | Approx | LS1 | LS2')
    parser.add_argument('-time', required=True, help='cutoff_in_seconds')
    parser.add_argument('-seed', default=0, help='random_seed')
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


