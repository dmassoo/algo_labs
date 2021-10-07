"""
This file is for experiment execution
"""


import sys



def main():
    from components import experiment as e, algorithms as a
    sys.setrecursionlimit(2000000)
    e.run_experiment1(a.const, 'const')
    e.run_experiment1(a.my_sum, 'sum')
    e.run_experiment1(a.product, 'product')
    e.run_experiment1(a.fixed_polynomial, 'polynomial')
    e.run_experiment1(a.fixed_horner, 'Horner')
    e.run_experiment1(a.bubble_sort, 'bubble_sort')
    e.run_experiment1(a.quick_sort, 'bubble_sort')
    e.run_experiment1(a.tim_sort, 'bubble_sort')

    e.run_experiment2(a.matrix_product, 'matrix_product')


main()
