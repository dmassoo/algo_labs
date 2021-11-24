import random
from matplotlib import pyplot as plt
from pprint import pprint
from time import time
import networkx as nx
import numpy as np


# Data generation
def generate_random_adjacency_matrix(v, e):
    """
    Generates n x n adjacency matrix with e edges for a simple undirected weighted graph
    """
    matrix = np.zeros([v, v])
    n_edges = 0
    edges = set()
    while n_edges != 2 * e:
        n = random.randint(0, v - 1)
        m = random.randint(0, v - 1)
        if frozenset([n, m]) not in edges and n != m:
            item = random.randint(1, 100)
            matrix[n][m] = matrix[m][n] = item
            edges.add(frozenset([n, m]))
            n_edges += 2
    return np.array(matrix)


# Floyd-Warshall
# Experiment
n = 10
TIMEUNIT = 1000
number_of_v = 50
G = None
min_span_tree = None
t = 0

densities = np.linspace(number_of_v, number_of_v*(number_of_v-1)/2, 10)
ts = []
nes = []
for nv in densities:
    t = 0
    for i in range(n):
        number_of_e = int(nv)
        matrix = generate_random_adjacency_matrix(number_of_v, number_of_e)
        G = nx.from_numpy_matrix(matrix)
        s = time()
        all_pairs = nx.floyd_warshall(G)
        delta = time() - s
        t += + delta
    t_n = TIMEUNIT * t / n
    print(f'Average all pairs shortest path finding time for the graph with {number_of_v} vertices and {number_of_e} edges: {t_n} ms')
    ts.append(t_n)
    nes.append(number_of_e)
print('T(E)')
print(ts)
print(nes)

plt.plot(nes, ts)
plt.scatter(nes,ts)
plt.xlabel('Number of edges')
plt.ylabel('Time, ms')
plt.show()


ts = []
nvs = []
for nv in list(range(100, 350, 50)):
    t = 0
    for i in range(n):
        number_of_e  = nv ** 2 / 5
        matrix = generate_random_adjacency_matrix(nv, number_of_e)
        G = nx.from_numpy_matrix(matrix)
        s = time()
        all_pairs = nx.floyd_warshall(G)
        delta = time() - s
        t += + delta
    t_n = TIMEUNIT * t / n
    print(f'Average all pairs shortest path finding time for the graph with {nv} vertices and {number_of_e} edges: {t_n} ms')
    ts.append(t_n)
    nvs.append(nv)
print('T(V)')
print(ts)
print(nvs)

plt.plot(nvs, ts)
plt.scatter(nvs, ts)
plt.xlabel('Number of vertices')
plt.ylabel('Time, ms')
plt.show()

# Visualization
matrix = generate_random_adjacency_matrix(50, 300)
print(matrix)
G = nx.from_numpy_matrix(matrix)
print('Number of nodes: ', G.number_of_nodes())
print('Number of edges: ', G.number_of_edges())
all_pairs = nx.floyd_warshall(G)
pprint(all_pairs)
layout = nx.spring_layout(G)
nx.draw_networkx(G, node_size=100, with_labels=True, pos=layout)
plt.show()