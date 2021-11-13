import random
from pprint import pprint
from time import process_time
import networkx as nx
import numpy as np


# Part I

# Data generation
def generate_random_adjacency_matrix(n, e):
    """
    Generates n x n adjacency matrix with e edges for a simple undirected weighted graph
    """
    matrix = np.zeros([n, n])
    n_edges = 0
    edges = set()
    while n_edges != 2 * e:
        n = random.randint(0, 99)
        m = random.randint(0, 99)
        if frozenset([n, m]) not in edges and n != m:
            item = random.randint(1, 100)
            matrix[n][m] = matrix[m][n] = item
            edges.add(frozenset([n, m]))
            n_edges += 2
    return np.array(matrix)


# Experiments
matrix = generate_random_adjacency_matrix(100, 500)
print(matrix)
G = nx.from_numpy_matrix(matrix)

# Dijkstra algorithm
num_of_measures = 10
random_node = random.randint(0, 99)
d_shortest_paths = {}
d_exec_times = []
for _ in range(num_of_measures):
    start_time = process_time()
    for node in G.nodes:
        if node != random_node:
            shortest_path = nx.dijkstra_path(G, random_node, node)
            d_shortest_paths.update({node: shortest_path})
    finish_time = process_time()
    d_exec_times.append(finish_time - start_time)

print(f"Random node is {random_node}")
print(f"Average time for Dijkstra's algorithm = {np.mean(d_exec_times)}")
print(f'The shortest path from node {random_node} to nodes: ')
pprint(d_shortest_paths)

# Bellman-Ford algorithm
bf_shortest_paths = {}
bf_exec_times = []
for _ in range(num_of_measures):
    start_time = process_time()
    for node in G.nodes:
        if node != random_node:
            shortest_path = nx.bellman_ford_path(G, random_node, node)
            bf_shortest_paths.update({node: shortest_path})
    finish_time = process_time()
    bf_exec_times.append(finish_time - start_time)

print(f"Random node is {random_node}")
print(f"Average time for Bellman-Ford algorithm = {np.mean(bf_exec_times)}")
print(f'The shortest path from node {random_node} to nodes: ')
pprint(bf_shortest_paths)
