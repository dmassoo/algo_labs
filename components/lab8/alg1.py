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


# Minimum Spanning Tree
# Experiment
n = 500
TIMEUNIT = 1000
number_of_v = 100
G = None
min_span_tree = None
t = 0

densities = np.linspace(number_of_v, number_of_v*(number_of_v-1)/2, 10)
ts = []
nes = []
for ne in densities:
    for i in range(n):
        number_of_e = int(ne)
        matrix = generate_random_adjacency_matrix(number_of_v, number_of_e)
        G = nx.from_numpy_matrix(matrix)
        s = time()
        min_span_tree = nx.minimum_spanning_tree(G)
        delta = time() - s
        t += + delta
    t_n = TIMEUNIT * t / n
    print(f'Average MST build time for the graph with {number_of_v} vertices and {number_of_e} edges: {t_n} ms')
    ts.append(t_n)
    nes.append(number_of_e)
print(ts)
print(nes)

plt.plot(nes, ts)
plt.scatter(nes, ts)
plt.xlabel('Number of edges')
plt.ylabel('Time, ms')
plt.show()

# Visualization
matrix = generate_random_adjacency_matrix(50, 300)
pprint(matrix)
G = nx.from_numpy_matrix(matrix)
min_span_tree = nx.minimum_spanning_tree(G)

pprint(sorted(min_span_tree.edges(data=True)))
plt.figure(figsize=(10, 10))
layout = nx.spring_layout(G)
nx.draw_networkx(G, pos=layout, node_size=number_of_v)
nx.draw_networkx(min_span_tree, pos=layout, node_size=number_of_v, edge_color="r")
plt.show()
