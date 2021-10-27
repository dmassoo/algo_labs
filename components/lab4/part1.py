import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt


#  Data Generation
def f(x):
    return 1 / (x ** 2 - 3 * x + 2)


def generate_data(f, x_k):
    y_k = list()
    d_k = np.random.normal(0, 1, 1001)
    for i in range(len(x_k)):
        if f(x_k[i]) < -10 ** 2:
            y_k.append(-10 ** 2 + d_k[i])
        elif f(x_k[i]) > 10 ** 2:
            y_k.append(10 ** 2 + d_k[i])
        else:
            y_k.append(f(3 * i / 1000))
    return y_k


x_k = [3 * i / 1000 for i in range(1001)]
y_k = generate_data(f, x_k)


def loss_func(x0):
    a0, b0, c0, d0 = x0
    D = 0
    for i in range(len(x_k)):
        D += (((a0 * x_k[i] + b0) / (x_k[i] ** 2 + c0 * x_k[i] + d0)) - y_k[i]) ** 2
    return D


# Optimisations
x0 = np.array([1, 1, 1, 1])

print('Nelder-Mead')
min_result_NM = spo.minimize(loss_func, x0, method='Nelder-Mead', options={'disp': True})
NM_a = list(min_result_NM.x)[0]
NM_b = list(min_result_NM.x)[1]
NM_c = list(min_result_NM.x)[2]
NM_d = list(min_result_NM.x)[3]

y_NM = []
for i in x_k:
    y_NM.append((NM_a * i + NM_b) / (i ** 2 + NM_c * i + NM_d))

print()

# LM
print('LM')
min_result_LM = spo.least_squares(loss_func, x0, gtol=0.001, verbose=1)
print(min_result_LM)
LM_a = list(min_result_LM.x)[0]
LM_b = list(min_result_LM.x)[1]
LM_c = list(min_result_LM.x)[2]
LM_d = list(min_result_LM.x)[3]
y_LM = []
for i in x_k:
    y_LM.append((LM_a * i + LM_b) / (i ** 2 + LM_c * i + LM_d))

print()

# Simulated Annealing
print('Simulated Annealing')

min_result_SA = spo.basinhopping(loss_func, x0, disp=True)
print(min_result_SA)
SA_a = list(min_result_SA.x)[0]
SA_b = list(min_result_SA.x)[1]
SA_c = list(min_result_SA.x)[2]
SA_d = list(min_result_SA.x)[3]
y_SA = []
for i in x_k:
    y_SA.append((SA_a * i + SA_b) / (i ** 2 + SA_c * i + SA_d))

print()

# Differential Evolution
print('Differential Evolution')
x0 = np.array([(1, 1), (1, 1), (1, 1), (1, 1)])
min_result_DE = spo.differential_evolution(loss_func, x0, disp=True)
print(min_result_DE)
DE_a = list(min_result_DE.x)[0]
DE_b = list(min_result_DE.x)[1]
DE_c = list(min_result_DE.x)[2]
DE_d = list(min_result_DE.x)[3]
y_DE = []
for i in x_k:
    y_DE.append((DE_a * i + DE_b) / (i ** 2 + DE_c * i + DE_d))

plt.scatter(x_k, y_k, label='Generated data', s=6)
plt.plot(x_k, y_SA, color='g', label='Simulated Annealing method', linewidth=6)
plt.plot(x_k, y_NM, color='r', label='Nelder-Mead method', linewidth=4)
plt.plot(x_k, y_DE, color='y', label='Differential Evolution method', linewidth=2)
plt.plot(x_k, y_LM, color='black', label='Levenberg-Marquardt method')
plt.legend()
plt.show()
