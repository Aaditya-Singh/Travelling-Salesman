## Import packages
import os
import math

## Read input tsv file into a dictionary
def TspToDict(Filepath):
	File = open(Filepath, 'r'); FileDict = {}
	Name = File.readline().strip().split()[1]; FileDict["Name"] = Name
	Comment = File.readline().strip().split()[1]; FileDict["Comment"] = Comment
	Dimension = int(File.readline().strip().split()[1]); FileDict["Dimension"] = Dimension
	EdgeType = File.readline().strip().split()[1]; FileDict["EdgeType"] = EdgeType
	File.readline()
	Nodes = [[0, 0] for k in range(Dimension)]
	for i in range(Dimension):
		NodeList = File.readline().strip().split()
		NodeID = int(NodeList[0]) - 1; NodeX = float(NodeList[1]); NodeY = float(NodeList[2])
		Nodes[NodeID] = [NodeX, NodeY]
	FileDict["Nodes"] = Nodes
	return FileDict

## Find distances between every two pair of points
def FindEdgeWeights(Dimension, NodeList):
	EdgeWeights = [[0 for j in range(Dimension)] for i in range(Dimension)]
	for i in range(Dimension):
		X1 = NodeList[i][0]; Y1 = NodeList[i][1]
		for j in range(Dimension):
			X2 = NodeList[j][0]; Y2 = NodeList[j][1]
			EdgeWeights[i][j] = round(math.sqrt(((X1 - X2)**2 + (Y1 - Y2)**2)))
	return EdgeWeights

## Explore all possible edges that have a total cost less than the current minimum
def backtrack(Index, Dimension, SumTillNow, MinSum, Visited, NodesTillNow, MinNodes, EdgeWeights):
	if MinSum[0] != -1 and SumTillNow >= MinSum[0]: return
	elif len(NodesTillNow) == Dimension:
		print("Current Sum: ", SumTillNow, end = " "); print("Min Sum: ", MinSum[0]); 
		print("Current Nodes: ", NodesTillNow)
		MinSum[0] = SumTillNow; MinNodes = NodesTillNow; return
	Visited[Index] = 1
	for Neighbor in range(Dimension):
		if Visited[Neighbor]: continue
		NodesTillNow.append(Neighbor)
		backtrack(Neighbor, Dimension, SumTillNow + EdgeWeights[Index][Neighbor], MinSum, \
			Visited, NodesTillNow, MinNodes, EdgeWeights)
		NodesTillNow.pop()
	Visited[Index] = 0
	
			
if __name__ == "__main__":
	Filepath = "./DATA/Atlanta.tsp"
	# print(Filepath)
	FileDict = TspToDict(Filepath)
	# print(FileDict["Nodes"])
	Dimension = FileDict["Dimension"]; Nodes = FileDict["Nodes"]
	EdgeWeights = FindEdgeWeights(Dimension, Nodes)
	# print(EdgeWeights[0])
	Visited = [0 for k in range(Dimension)]
	NodesTillNow = []; MinNodes = []; MinSum = [-1]
	for Index in range(Dimension):
		NodesTillNow.append(Index)
		backtrack(Index, Dimension, 0, MinSum, Visited, NodesTillNow, MinNodes, EdgeWeights)
		NodesTillNow.pop()
	print(MinNodes)
	


