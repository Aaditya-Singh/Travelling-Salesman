import ACO_Class as aco

obj = aco.AntColonyOptimization(Filepath="./DATA/Atlanta.tsp", threshold_time=10)
obj.RUN_ACO()
