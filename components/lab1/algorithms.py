# fixed value for polynomials
X = 1.5


def const(v: list) -> int:
    """
    Returns constant
    """
    return 0


def my_sum(v: list) -> int:
    """
    Returns sum of the vector elements
    """
    return sum(v)

def product(v: list) -> int:
    """
    Returns product of the vector elements
    """
    p = 1
    for i in v:
        p *= i
    return p


def fixed_polynomial(v: list) -> float:
    return polynomial(v, X)


def polynomial(v: list, x: float) -> float:
    """
    Considering v as a list of coefficients, calculates its value in X
    """
    x_powers = [x ** k for k in range(len(v))]
    return sum([v_i * x_i for v_i, x_i in zip(v, x_powers)])


def fixed_horner(v: list) -> float:
    return horner(v, X)


def horner(v: list, x: float) -> float:
    """
    Considering v as a list of coefficients, calculates its value in X using Horner's method
    """

    def _horner(v: list, x: float) -> float:
        if len(v) <= 1:
            return v[0] + x
        else:
            return v[0] + x * _horner(v[1::], x)

    return _horner(v, x)


def bubble_sort(v: list) -> list:
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(v) - 1):
            if v[i] > v[i + 1]:
                # Swap the elements
                v[i], v[i + 1] = v[i + 1], v[i]
                swapped = True
    return v


def quick_sort(v: list) -> list:
    def _quick_sort(v, low, high):
        if low < high:
            split_index = partition(v, low, high)
            _quick_sort(v, low, split_index)
            _quick_sort(v, split_index + 1, high)

    _quick_sort(v, 0, len(v) - 1)
    return v


def partition(v: list, low: int, high: int):
    pivot = v[(low + high) // 2]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while v[i] < pivot:
            i += 1

        j -= 1
        while v[j] > pivot:
            j -= 1

        if i >= j:
            return j
        v[i], v[j] = v[j], v[i]


def tim_sort(v: list) -> list:
    return sorted(v)


def matrix_product(a: list, b: list) -> list:
    """
    Matrix product
    Returns empty list if matrices cannot be multiplied
    """
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])

    if cols_a != rows_b:
        return []

    c = [[0 for row in range(cols_b)] for col in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                c[i][j] = a[i][k] * b[k][j]

    return c
