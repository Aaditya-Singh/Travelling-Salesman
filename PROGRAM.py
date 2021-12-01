## Import packages
import os
import time
import math

## Class for branch and bound algorithm
class BranchAndBound():
	def __init__(self, Filepath, Cutoff):
		self.Filepath = Filepath
		self.FileDict = {}
		self.Dimension = 0
		self.Nodes = []
		self.EdgeWeights = []
		self.ClosestPoints = []
		self.DistToNodes = {}
		self.SortedEdgeWeights = []
		self.PathToWeight = {}
		self.MinWeights = []
		self.BestPaths = []
		self.BestTraces = []
		self.StartTime = time.time()
		self.EndTime = self.StartTime + Cutoff
	
	## Read input tsv file into a dictionary
	def TspToDict(self):
		File = open(self.Filepath, 'r')
		Name = File.readline().strip().split()[1]; self.FileDict["Name"] = Name
		Comment = File.readline().strip().split()[1]; self.FileDict["Comment"] = Comment
		Dimension = int(File.readline().strip().split()[1])
		self.FileDict["Dimension"] = Dimension; self.Dimension = Dimension
		EdgeType = File.readline().strip().split()[1]; self.FileDict["EdgeType"] = EdgeType
		File.readline()
		self.Nodes = [[0, 0] for k in range(self.Dimension)]
		for i in range(Dimension):
			NodeList = File.readline().strip().split()
			NodeID = int(NodeList[0]) - 1
			NodeX = float(NodeList[1]); NodeY = float(NodeList[2])
			self.Nodes[NodeID] = [NodeX, NodeY]
		self.FileDict["Nodes"] = self.Nodes

	## Find distances between every two pair of points
	def FindEdgeWeightsAndClosestPoints(self):
		self.EdgeWeights = [[int(1e+9) for j in range(self.Dimension)] for i in range(self.Dimension)]
		self.ClosestPoints = [[0 for j in range(self.Dimension)] for i in range(self.Dimension)]
		for i in range(self.Dimension):
			X1 = self.Nodes[i][0]; Y1 = self.Nodes[i][1]
			for j in range(self.Dimension):
				if i == j: continue
				X2 = self.Nodes[j][0]; Y2 = self.Nodes[j][1]
				self.EdgeWeights[i][j] = round(math.sqrt(((X1 - X2)**2 + (Y1 - Y2)**2)))
			self.ClosestPoints[i] = sorted(range(len(self.EdgeWeights[i])), \
				key = self.EdgeWeights[i].__getitem__)

	## Map distances to pair of points and sort the distances
	def MapAndSortEdgeWeights(self):
		for i in range(self.Dimension):
			for j in range(i + 1, self.Dimension):
				distance = self.EdgeWeights[i][j]
				self.DistToNodes[distance] = [i, j]
				self.SortedEdgeWeights.append(distance)
		self.SortedEdgeWeights.sort()

	## Kruskal's algorithm to find total weight of a minimum spanning tree
	def FindLowerBound(self, PartialPath):
		SortedPartialPath = sorted(PartialPath[1: -1])
		SortedEdgeWeightsCopy = self.SortedEdgeWeights.copy()
		ListOfNodes = [str(k) for k in SortedPartialPath]
		StringOfNodes = "".join(ListOfNodes)
		if StringOfNodes in self.PathToWeight: return self.PathToWeight[StringOfNodes]
		# Unseen partial path	
		Ancestors = {}
		for i in range(self.Dimension):
			if i not in PartialPath[1: -1]: Ancestors[i] = i 
		TotalWeight = 0
		while(len(SortedEdgeWeightsCopy)):
			CurrEdge = SortedEdgeWeightsCopy.pop(0)
			X = self.DistToNodes[CurrEdge][0]; Y = self.DistToNodes[CurrEdge][1]
			if X in PartialPath[1: -1] or Y in PartialPath[1: -1] or Ancestors[X] == Ancestors[Y]: 
				continue
			Z = Ancestors[Y]
			for Node in Ancestors.keys():
				if Ancestors[Node] == Z: Ancestors[Node] = Ancestors[X]
			TotalWeight += CurrEdge
		self.PathToWeight[StringOfNodes] = TotalWeight
		return TotalWeight

	## Find optimal path in a given time limit
	def FindOptimalPath(self, CurrWeight, OptWeight, PartialPath):
		if(time.time() > self.EndTime) : return
		elif len(PartialPath) == self.Dimension: 
			X = PartialPath[-1]; Y = PartialPath[0]
			CurrWeight += self.EdgeWeights[X][Y]
			if not self.MinWeights or CurrWeight < self.MinWeights[-1]: 
				self.MinWeights.append(CurrWeight); self.BestPaths.append(PartialPath.copy())
				self.BestTraces.append([round(time.time() - self.StartTime, 2), int(CurrWeight)])
				print(self.MinWeights[-1])
				print(self.BestPaths[-1])
				print(self.BestTraces[-1])
			return
		i = PartialPath[-1]
		for j in self.ClosestPoints[i]:
			if j in PartialPath: continue
			PartialPath.append(j)
			LowerBound = self.FindLowerBound(PartialPath)
			if self.MinWeights and LowerBound >= self.MinWeights[-1]: continue
			self.FindOptimalPath(CurrWeight + self.EdgeWeights[i][j], OptWeight, PartialPath)
			PartialPath.pop()
	
	## Call the appropriate functions in order and construct optimal path
	def Main(self, OptWeight=int(1e+9)):
		self.TspToDict()
		self.FindEdgeWeightsAndClosestPoints()
		self.MapAndSortEdgeWeights()		
		PartialPath = []
		for i in range(self.Dimension):
			CurrWeight = 0
			PartialPath.append(i)
			self.FindOptimalPath(CurrWeight, OptWeight, PartialPath)
			PartialPath.pop()
