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
    min_val = func(a)
    c = a
    f_c = 0
    iters = 0
    while c <= b:
        iters += 1
        f_c = func(c)
        if f_c < min_val:
            min_val = f_c
        c += (b - a) / n
    return c, f_c, iters, iters


def dichotomy(func, interval: (float, float), eps: float) -> (float, float, int, int):
    check_params(interval, eps)
    a, b = interval
    d = eps / 5
    c = 0
    iters = 0
    calcs = 0
    f_min = 0
    while abs(b - a) >= eps:
        iters += 1

        c = (b - a) / 2
        y1 = func(c - d)
        y2 = func(c + d)
        calcs += 2

        if y1 < y2:
            b = c
            f_min = y1
        else:
            a = c
            f_min = y2
    return c, f_min, calcs, iters


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
