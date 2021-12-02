import matplotlib.pyplot as plt
import numpy as np

THEORY = 'Theoretical estimates'
EXPERIMENT = 'Experimental results'
SIZE = 'Data size'
TIME_MICROSECONDS = 'Time, microseconds'
nameTofunc = {"const": 0,
              "sum": 1,
              "product": 1,
              "polynomial": 'nlogn',
              "horner": 1,
              "bubble_sort": 2,
              "quick_sort": 'nlogn',
              "matrix_product": 3}


def converter(cell):
    return float(str.replace(cell.decode('ascii'), ' ', ''))

# weird plotting code

# const
data = np.loadtxt(f'data/const.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = [np.polyfit(n, t, 0)] * data_size
plt.plot(approx, label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)

plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Const function', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# sum
data = np.loadtxt(f'data/sum.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 1))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Sum function', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# product
data = np.loadtxt(f'data/product.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 1))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Product function', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# polynomial
data = np.loadtxt(f'data/polynomial.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 2))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Polynomial function', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# horner
data = np.loadtxt(f'data/horner.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 1))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Horner function', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# bubble_sort
data = np.loadtxt(f'data/bubble_sort.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 2))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Bubble sort', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# quick_sort
data = np.loadtxt(f'data/quick_sort.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 2))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Quick sort', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# tim_sort
data = np.loadtxt(f'data/tim_sort.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 2))
plt.plot(approx(n), label=THEORY)
plt.plot(t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Tim sort', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()

# matrix_product
data = np.loadtxt(f'data/matrix_product.csv', delimiter=' , ', converters={0: converter, 1: converter})
n = data[:, 0]
t = data[:, 1]
data_size = np.size(t)
approx = np.poly1d(np.polyfit(n, t, 3))
xs = range(1, 300, 10)
plt.plot(xs, approx(n), label=THEORY)
plt.plot(xs, t, label=EXPERIMENT, alpha=0.7)

plt.xlabel(SIZE, fontsize=16)
plt.ylabel(TIME_MICROSECONDS, fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Matrix product', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.show()
