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

    adj_list = []
    for row in mat:
        adj_list.append(list_for_node_row_n(row))
    return adj_list


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
print('dfs: ', list(nx.dfs_edges(G, source=0)))
print('bfs: ', list(nx.bfs_edges(G, source=0)))

# Visualization
nx.draw(G, node_size=100, with_labels=True, )
plt.show()
