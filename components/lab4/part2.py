import pandas as pd
import components.lab4.sa as sa

df = pd.read_csv('adjacency_matrix.csv', sep=',', header=None)
sa.sa_algorithm(df)
