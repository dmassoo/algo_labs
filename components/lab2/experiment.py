import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
from components.lab2.algorithms import *

# Part I
int_1 = int_2 = (0, 1)
int_3 = (0.01, 1)
eps = 0.001


print("exhaustive_search")
print("x_min, y_min, calculations, iterations")
print(exhaustive_search(f1, int_1, eps))
print(exhaustive_search(f2, int_2, eps))
print(exhaustive_search(f3, int_3, eps))


print("dichotomy")
print("x_min, y_min, calculations, iterations")
print(dichotomy(f1, int_1, eps))
print(dichotomy(f2, int_2, eps))
print(dichotomy(f3, int_3, eps))

print("golden_section")
print("x_min, y_min, calculations, iterations")
print(golden_section(f1, int_1, eps))
print(golden_section(f2, int_2, eps))
print(golden_section(f3, int_3, eps))


# Part II
def linear_loss(a_b):
    a, b = a_b
    D = 0
    for i in range(len(X)):
        D += (f4(X[i], a, b) - Y[i]) ** 2
    return D


def rational_loss(a_b):
    a, b = a_b
    D = 0
    for i in range(len(X)):
        D += (f5(X[i], a, b) - Y[i]) ** 2
    return D


def generate_data_by_func(func, X, a_b):
    a, b = a_b
    return [func(x_k, a, b) for x_k in X]


alpha, beta = generate_random_coeffs()

X, Y = generate_data(alpha, beta)
x_0 = np.array([1, 1])

# Linear
Y_generative_line = generative_line(X, alpha, beta)

min_result_ES_l = scipy.optimize.brute(linear_loss, ((-1, 1), (-1, 1)), disp=True)


min_result_NM = scipy.optimize.minimize(linear_loss, x_0, method='Nelder-Mead', options={'disp': True})

min_result_CD = scipy.optimize.minimize(linear_loss, x_0, method='Powell', options={'disp': True})


# Rational
Y_generative_line_rational = generative_line_rational(X, alpha, beta)

min_result_ES_r = scipy.optimize.brute(rational_loss, ((-1, 1), (-1, 1)), disp=True)

min_result_NM_r = scipy.optimize.minimize(rational_loss, x_0, method='Nelder-Mead', options={'disp': True})

min_result_CD_r = scipy.optimize.minimize(rational_loss, x_0, method='Powell', options={'disp': True})


# Data Preparation for visualization
Y_ES_l = generate_data_by_func(f4, X, min_result_ES_l)
Y_NM_l = generate_data_by_func(f4, X, min_result_NM.x)
Y_CD_l = generate_data_by_func(f4, X, min_result_CD.x)

Y_ES_r = generate_data_by_func(f5, X, min_result_ES_r)
Y_NM_r = generate_data_by_func(f5, X, min_result_NM_r.x)
Y_CD_r = generate_data_by_func(f5, X, min_result_CD_r.x)

# Visualization

# Linear
plt.scatter(X, Y, color='b', label='Generated data')
plt.plot(X, Y_generative_line, color='g', label='Generative line')
plt.plot(X, Y_ES_l, color='orange', label='Exhaustive Search linear approximant')
plt.plot(X, Y_NM_l, color='r', label='Nelder-Mead linear approximant')
plt.plot(X, Y_CD_l, color='y', label='Coordinate Descent linear approximant')
plt.legend()
plt.show()


# Rational
plt.scatter(X, Y, color='b', label='Generated data')
plt.plot(X, Y_generative_line_rational, color='g', label='Generative line')
plt.plot(X, Y_ES_r, color='orange', label='Exhaustive Search rational approximant')
plt.plot(X, Y_NM_r, color='r', label='Nelder-Mead rational approximant')
plt.plot(X, Y_CD_r, color='y', label='Coordinate Descent rational approximant')
plt.legend()
plt.show()