import random as rd
import numpy.random as nrd
from tqdm import tqdm


#   Part I

def f1(x):
    return x ** 3


def f2(x):
    return abs(x - 0.2)


def f3(x):
    from math import sin
    return x * sin(1 / x)


def exhaustive_search(func, interval: (float, float), eps: float) -> (float, float, int, int):
    check_params(interval, eps)
    a, b = interval
    n = (b - a) / eps
    x_min = a
    y_min = func(a)
    c = a
    iters = 0
    while c <= b:
        iters += 1
        f_c = func(c)
        if f_c < y_min:
            y_min = f_c
            x_min = c
        c += (b - a) / n
    return x_min, y_min, iters, iters


def dichotomy(func, interval: (float, float), eps: float) -> (float, float, int, int):
    check_params(interval, eps)
    a, b = interval
    d = eps / 2
    c = 0
    iters = 0
    calcs = 0
    while abs(b - a) >= eps:
        iters += 1
        c = (a + b) / 2
        y1 = func(c - d)
        y2 = func(c + d)
        calcs += 2

        if y1 < y2:
            b = c
        else:
            a = c
    return c, func(c), calcs, iters


def golden_section(func, interval: (float, float), eps: float) -> (float, float, int, int):
    phi = (3 - 5 ** 0.5)
    check_params(interval, eps)
    a, b = interval

    c = (b - a) / 2
    x1 = a + c * phi
    x2 = a - c * phi
    y1 = func(x1)
    y2 = func(x2)
    iters = 1
    calcs = 2

    while abs(b - a) >= eps:
        if y1 <= y2:
            b = x2
            x2 = x1
            y2 = y1
            x1 = (b - a) / 2 * phi + a
            y1 = func(x1)
        else:
            a = x1
            x1 = x2
            y1 = y2
            x2 = - (b - a) / 2 * phi + a
            y2 = func(x2)
        iters += 1
        calcs += 1
    x_m = abs(a - b) / 2
    return x_m, func(x_m), calcs, iters


def check_params(interval: (float, float), eps: float):
    a, b = interval
    assert b > a
    assert eps > 0


#     Part II

# Data Generation and Functions definition

def generate_random_coeffs():
    rd.seed(1000)
    return rd.random(), rd.random()


def generate_data(alpha, beta):
    deltas = nrd.normal(0, 1, 101)
    X = [k / 100 for k in range(101)]
    Y = [alpha * X[k] + beta + deltas[k] for k in range(101)]
    return X, Y


def f4(x, a, b):
    return a * x + b


def f5(x, a, b):
    return a / (1 + b * x)


def generative_line(X, a, b):
    return [a * x_k + b for x_k in X]


def generative_line_rational(X, a, b):
    return [a / (1 + b * x_k) for x_k in X]
