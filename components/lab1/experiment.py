import csv
from time import time
"""
Experiment constants and benchmarking methods
"""
MAX_RANDOM_INT = 100
MAX_DATA_SIZE = 2000
STEP = 10
NUMBER_OF_RUNS = 5
MICROSECONDS_MULTIPLIER = 10e6
SCALE = 4


def run_experiment1(func, filename):
    with open(f'data/{filename}.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in range(1, MAX_DATA_SIZE):
            row = f'{i},{experiment1_fixed_n(i, NUMBER_OF_RUNS, func)}'
            writer.writerow(row)


def experiment1_fixed_n(data_size: int, reps: int, func) -> float:
    vectors = [generate_random_vector(data_size) for i in range(reps)]
    total_time = 0
    for vector in vectors:
        start = time()
        func(vector)
        end = time()
        total_time += (end - start)
    avg_time = total_time / reps
    return round(avg_time * MICROSECONDS_MULTIPLIER, SCALE)


def run_experiment2(func, filename):
    with open(f'data/{filename}.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        for i in range(1, MAX_DATA_SIZE, STEP):
            row = f'{i},{experiment2_fixed_n(i, NUMBER_OF_RUNS, func)}'
            writer.writerow(row)


def experiment2_fixed_n(data_size: int, reps: int, func) -> float:
    matrix_pairs = [(generate_random_matrix(data_size), generate_random_matrix(data_size)) for i in range(reps)]
    total_time = 0
    for pair in matrix_pairs:
        start = time()
        func(*pair)
        end = time()
        total_time += (end - start)
    avg_time = total_time / reps
    return round(avg_time * MICROSECONDS_MULTIPLIER, SCALE)


def generate_random_vector(n: int) -> list:
    from random import randint
    return [abs(randint(0, MAX_RANDOM_INT)) for i in range(n)]


def generate_random_matrix(n: int) -> list:
    return [generate_random_vector(n) for i in range(n)]
