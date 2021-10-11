"""
This file is for experiment execution
There are two ways to conduct:
1) One by one ---- main()
2) All the algorithms in one run ---- main_all()
"""

import sys
from components.lab1 import algorithms as a, experiment as e

sys.setrecursionlimit(2000000)

nameTofunc = {"const": a.const,
              "sum": a.my_sum,
              "product": a.product,
              "polynomial": a.fixed_polynomial,
              "horner": a.fixed_horner,
              "bubble_sort": a.bubble_sort,
              "quick_sort": a.quick_sort,
              "matrix_product": a.matrix_product}


def main():
    print(f"Choose a function to benchmark: {nameTofunc.keys()} or matrix_product.")
    func_name = input("Type function name to start:")
    if func_name == "matrix_product":
        e.run_experiment2(nameTofunc.get(func_name), func_name)
    elif func_name in nameTofunc.keys():
        e.run_experiment1(nameTofunc.get(func_name), func_name)
    print(f'{func_name} is benchmarked')


def main_all():
    e.run_experiment1(a.const, 'const')
    e.run_experiment1(a.my_sum, 'sum')
    e.run_experiment1(a.product, 'product')
    e.run_experiment1(a.fixed_polynomial, 'polynomial')
    e.run_experiment1(a.fixed_horner, 'Horner')
    e.run_experiment1(a.bubble_sort, 'bubble_sort')
    e.run_experiment1(a.quick_sort, 'quick_sort')
    e.run_experiment1(a.tim_sort, 'tim_sort')

    e.run_experiment2(a.matrix_product, 'matrix_product')


# main()
