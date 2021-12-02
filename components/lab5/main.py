import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Matrix and List generation
def generate_matrix(v, e):
    coords = set()
    matrix = [[0 for _ in range(v)] for _ in range(v)]
    while len(coords) < e:
        i = np.random.randint(0, v)
        j = np.random.randint(i, v)
        if i != j:
            coords.add((i, j))
            matrix[i][j] = matrix[j][i] = 1
    return np.array(matrix)


def mat_to_list(mat):
    def list_for_node_row_n(row):
        l = []
        for i in range(len(row)):
            if row[i] == 1:
                l.append(i)
        return l

    adj_list = {}
    for i in range(len(mat)):
        adj_list[i] = list_for_node_row_n(mat[i])
    return adj_list


def bfs_shortest_path(graph, start, goal):
    explored = []

    queue = [[start]]

    if start == goal:
        print("Same Node")
        return

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in explored:
            neighbours = graph[node]

            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour == goal:
                    print("Shortest path = ", *new_path)
                    return
            explored.append(node)

    print("Connecting path doesn't exist")


# Adjacency matrix and list for fixed 100 x 100 matrix with 200 edges
adj_mat = generate_matrix(100, 200)
adj_list = mat_to_list(adj_mat)
print(adj_mat)
print(adj_list)

# Graph
# Connected components, dfs and bfs
G = nx.from_numpy_matrix(adj_mat)
print('dfs connected components: ', list(nx.connected_components(G)))
print('dfs number connected components: ', nx.number_connected_components(G))

start = np.random.randint(0, 99)
target = np.random.randint(start, 99)
print(f'Starting from edge {start} to {target}')
bfs_shortest_path(adj_list, start, target)

# Visualization
nx.draw(G, node_size=100, with_labels=True)
plt.show()
