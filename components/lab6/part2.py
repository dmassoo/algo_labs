import random
from matplotlib import pyplot as plt
import networkx as nx


def choose_cells():
    """
    Returns two different non-obstacle random cells from the grid
    """
    flag = True
    while flag:
        start = (random.randint(0, 19), random.randint(0, 9))
        finish = (random.randint(0, 19), random.randint(0, 9))
        if start not in obstacles and finish not in obstacles and start != finish:
            print('start: ', start)
            print('finish: ', finish)
            flag = False
    return start, finish


# Grid without obstacles
G = nx.grid_2d_graph(20, 10)


# Adding obstacles
counter = 0
obstacles = []
while counter != 30:
    block = (random.randint(0, 19), random.randint(0, 9))
    if block not in obstacles:
        obstacles.append(block)
        G.remove_node(block)
        counter += 1
print('obstacles: ', obstacles)


# Visualization
weights = nx.get_edge_attributes(G, 'weight')
pos = dict((n, n) for n in G.nodes())

plt.figure(figsize=(15, 15))
nx.draw_networkx(G, pos=pos, node_size=100)
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

for _ in range(5):
    start, finish = choose_cells()
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(G, pos=pos, node_size=100)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    path = nx.astar_path(G, start, finish)
    edges = [(a, b) for a, b in zip(path, path[1:])]
    print('edges: ', edges)
    nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="r", width=3)
    plt.show()
