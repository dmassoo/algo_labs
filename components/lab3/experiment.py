import random as rd
import numpy as np
import scipy.optimize as spo

eps = 0.001


def generate_random_coeffs():
    rd.seed(1000)
    return rd.random(), rd.random()


def generate_data(alpha, beta):
    deltas = np.random.normal(0, 1, 101)
    X = [k / 100 for k in range(101)]
    Y = [alpha * X[k] + beta + deltas[k] for k in range(101)]
    return np.array(X), np.array(Y)


def f4(x, a, b):
    return a * x + b


def f5(x, a, b):
    return a / (1 + b * x)


def generative_line(func, X, a, b):
    return [func(x_k, a, b) for x_k in X]


def linear_loss(a_b):
    a, b = a_b
    D = 0
    for i in range(len(X)):
        D += (f4(X[i], a, b) - Y[i]) ** 2
    return D


def linear_loss_lm(a_b):
    a, b = a_b
    global X, Y, f4
    return (f4(X, a, b) - Y)


def rational_loss(a_b):
    a, b = a_b
    D = 0
    for i in range(len(X)):
        D += (f5(X[i], a, b) - Y[i]) ** 2
    return D


def rational_loss_lm(a_b):
    a, b = a_b
    global X, Y
    return f5(X, a, b) - Y


def generate_data_by_func(func, X, a_b):
    a, b = a_b
    return [func(x_k, a, b) for x_k in X]


# Data generation
alpha, beta = generate_random_coeffs()

X, Y = generate_data(alpha, beta)
x_0 = np.array([1.0, 1.0])


# funcs for gradient calculation
def jac_linear(alpha_betta):
    a1, b1 = alpha_betta
    n_points = len(X)
    a_grad_linear = 0
    b_grad_linear = 0
    for i in range(n_points):
        da_linear = -((2 / n_points) * X[i] * (Y[i] - (a1 * X[i] + b1)))
        db_linear = - ((2 / n_points) * (Y[i] - (a1 * X[i] + b1)))
        a_grad_linear = a_grad_linear + da_linear
        b_grad_linear = b_grad_linear + db_linear
    return np.array([a_grad_linear, b_grad_linear])


def jac_rational(alpha_betta):
    a2, b2 = alpha_betta
    n_points = len(X)
    a_grad_rat = 0
    b_grad_rat = 0
    for i in range(n_points):
        da_rational = -(2 * (-a2 + Y[i] + b2 * X[i] * Y[i])) / (((1 + b2 * X[i]) ** 2) * n_points)
        db_rational = -(2 * a2 * X[i] * (a2 - Y[i] * (1 + X[i] * b2))) / (((1 + X[i] * b2) ** 3) * n_points)
        a_grad_rat = a_grad_rat + da_rational
        b_grad_rat = b_grad_rat + db_rational
    return np.array([a_grad_rat, b_grad_rat])


def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = np.copy(start)
    iteration = 0
    for _ in range(n_iter):
        iteration += 1
        diff = -learn_rate * gradient(vector)
        print(diff.shape)
        if np.all(np.abs(diff) <= tolerance):
            print("Criterion stop, iterations = ", iteration)
            break
        vector += diff
    return vector


def gradient_descent_step(a1, b1, a2, b2):
    learning_rate = 0.0001
    # Calculation of derivatives
    da_linear, db_linear = jac_linear((a1, b1))
    da_rational, db_rational = jac_rational((a2, b2))
    # Step
    a_updated_lin = a1 - learning_rate * da_linear
    b_updated_lin = b1 - learning_rate * db_linear
    a_updated_rat = a2 - learning_rate * da_rational
    b_updated_rat = b2 - learning_rate * db_rational
    return a_updated_lin, b_updated_lin, a_updated_rat, b_updated_rat

# Gradient Descent
# Initial guess for both linear and rational losses
alpha_lin, betta_lin, alpha_rat, betta_rat = 1, 1, 1, 1
for i in range(200000):
    alpha_lin, betta_lin, alpha_rat, betta_rat = gradient_descent_step(alpha_lin, betta_lin, alpha_rat, betta_rat)
    y_plot_lin = []
    y_plot_rat = []
    for x in X:
        y_plot_lin.append(alpha_lin * x + betta_lin)
        y_plot_rat.append(alpha_rat / (1 + x * betta_rat))

# Linear
Y_generative_line = generative_line(f4, X, alpha, beta)
cg_linear = spo.fmin_cg(linear_loss, x_0, gtol=0.001, disp=True)
ncg_linear = spo.minimize(linear_loss, x_0, jac=jac_linear, method='BFGS', options={'disp': True})
lm_linear = spo.least_squares(linear_loss_lm, x_0, method='lm', ftol=eps, xtol=eps, verbose=1)

# plotting data
Y_GD_l = y_plot_lin
Y_CG_l = generate_data_by_func(f4, X, cg_linear)
Y_NCG_l = generate_data_by_func(f4, X, ncg_linear.x)
Y_LM_l = generate_data_by_func(f4, X, lm_linear.x)

# Rational
cg_rational = spo.fmin_cg(rational_loss, x_0, gtol=0.001, disp=True)
# cg_rational = spo.minimize(rational_loss, x_0, method='CG', gtol=eps, options={'disp': True})
ncg_rational = spo.minimize(rational_loss, x_0, jac=jac_rational, method='BFGS', options={'disp': True})
lm_rational = spo.least_squares(rational_loss_lm, x_0, method='lm', ftol=eps, xtol=eps, verbose=1)

# plotting data
Y_GD_r = y_plot_rat
Y_CG_r = generate_data_by_func(f5, X, cg_rational)
Y_NCG_r = generate_data_by_func(f5, X, ncg_rational.x)
Y_LM_r = generate_data_by_func(f5, X, lm_rational.x)

# Visualization
from matplotlib import pyplot as plt

plt.scatter(X, Y, color='b', label='Generated data')
plt.plot(X, Y_generative_line, color='b', label='Generative line', linewidth=1)
plt.plot(X, Y_CG_l, color='y', label='Conjugate gradient descent linear', linewidth=6)
plt.plot(X, Y_NCG_l, color='orange', label='Newton linear', linewidth=4)
plt.plot(X, Y_LM_l, color='r', label='LM linear', linewidth=1)
plt.plot(X, Y_GD_l, color='black', label='Gradient descent linear')
plt.legend()
plt.show()

plt.scatter(X, Y, color='b', label='Generated data')
plt.plot(X, Y_generative_line, color='b', label='Generative line')
plt.plot(X, Y_CG_r, color='y', label='Conjugate gradient descent rational', linewidth=6)
plt.plot(X, Y_NCG_r, color='orange', label='Newton rational', linewidth=4)
plt.plot(X, Y_LM_r, color='r', label='LM rational', linewidth=1)
plt.plot(X, Y_GD_r, color='black', label='Gradient descent rational')
plt.legend()
plt.show()
