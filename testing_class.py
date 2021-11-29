import ACO_Class as aco

obj = aco.AntColonyOptimization(Filepath="./DATA/Cincinnati.tsp", threshold_time=10)
obj.RUN_ACO()