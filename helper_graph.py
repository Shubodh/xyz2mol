import sys
import copy


# Function to insert vertices
# to adjacency list
def insert(adj, u, v):

	# Insert a vertex v to vertex u
	adj[u].append(v)
	return

def insert_as_list(adj, uv):
	#uv is list: [u,v]
	u,v = uv

	# Insert a vertex v to vertex u
	adj[u].append(v)
	return

# Function to display adjacency list
def printList(adj, V):
	
	for i in range(V):
		print(i, end = '')
		
		for j in adj[i]:
			print(' --> ' + str(j), end = '')
			
		print()
		
	print()
		
# Function to convert adjacency
# list to adjacency matrix
def convert_adjlist_to_mat(adj, V, make_symmetric=False):

	# Initialize a matrix
	matrix = [[0 for j in range(V)]
				for i in range(V)]
	
	for i in range(V):
		for j in adj[i]:
			matrix[i][j] = 1
			if make_symmetric:
				matrix[j][i] = 1
	
	return matrix

def convert_adjMat_to_BOmat(adjMatrix, adjList_withBO, adjList, make_symmetric=False): #adjList_withBO is BOadj
	dict_edge_BOval = (list(adjList_withBO.values())[0])
	# print(adjMatrix)
	BOmat = copy.deepcopy(adjMatrix)
	for i in range(len(BOmat)):
		for j in adjList[i]:
			BOmat[i][j] = dict_edge_BOval[frozenset({i, j})] #HERE CURRENTLY
			# print(dict_edge_BOval[frozenset({52, 0})])
			if make_symmetric:
				BOmat[j][i] = dict_edge_BOval[frozenset({i, j})]
	# print("\n BOmat \n", BOmat, "\n adjMatrix \n", adjMatrix)
	return BOmat, adjMatrix 

# Function to display adjacency matrix
def printMatrix(adj, V):
	
	for i in range(V):
		for j in range(V):
			print(adj[i][j], end = ' ')
			
		print()
		
	print()

def example_usage():
	V = 5

	adjList = [[] for i in range(V)]

	# Inserting edges
	insert(adjList, 0, 1)
	insert(adjList, 0, 4)
	insert(adjList, 1, 0)
	insert(adjList, 1, 2)
	insert(adjList, 1, 3)
	insert(adjList, 1, 4)
	insert(adjList, 2, 1)
	insert(adjList, 2, 3)
	insert(adjList, 3, 1)
	insert(adjList, 3, 2)
	insert(adjList, 3, 4)
	insert(adjList, 4, 0)
	insert(adjList, 4, 1)
	insert(adjList, 4, 3)

	# Display adjacency list
	print("Adjacency List: ")
	print(adjList)
	printList(adjList, V)

	# Function call which returns
	# adjacency matrix after conversion
	adjMatrix = convert_adjlist_to_mat(adjList, V)

	# Display adjacency matrix
	print("Adjacency Matrix: ")
	printMatrix(adjMatrix, V)

def BOadj_to_BOmat(BOadj):
	dict_edge_BOval = (list(BOadj.values())[0])
	edge_list = list(dict_edge_BOval.keys())

	V = 55

	adjList = [[] for i in range(V)]

	for i_edge in edge_list:
		uv = (list(i_edge))
		insert_as_list(adjList, uv)
	

	# Function call which returns
	# adjacency matrix after conversion
	adjMatrix = convert_adjlist_to_mat(adjList, V, make_symmetric=True)


	BOmat, adjMatrix = convert_adjMat_to_BOmat(adjMatrix, BOadj, adjList, make_symmetric=True)
	# Display adjacency list
	debug = False
	if debug:
		print("Adjacency List: ")
		print(adjList)
		printList(adjList, V)
		# Display adjacency matrix
		print("Adjacency Matrix: ")
		printMatrix(adjMatrix, V)
		print(adjMatrix, "\n", BOmat)

	return BOmat, adjMatrix

		
if __name__=='__main__':
	print("example usage")
	example_usage()
