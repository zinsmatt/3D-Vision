import numpy as np
import matplotlib.pyplot as plt


""" === Polynomial Regression ===
It is possible to regress a non-linear fonction using polynomial regression.
=> This way the model is still linear wrt. the parameters that we want to
estimate.
=> The cost function is again a quadratic function.
=> A singled solution exist and can be obtained with the linear least-square
solution (closed-form).
"""


def f(x, a):
    return a * x**3 - 2 * a * x**2 + x

a = 1.35
N = 100
x = np.linspace(0, 1.5, N)
x2 = np.linspace(0, 1.5, N*5)
noise = np.random.randn(N) * 1e-2
y = f(x, a) + noise

plt.scatter(x, y, c="blue", s=1)


X = x**3 - 2 * x**2
a_est = 1.0 / X.dot(X) * X.dot(y-x)

y_filt = f(x2, a_est)
plt.scatter(x2, y_filt, c="red", s=1)



M = 100
a_ests = np.linspace(-5, 5, M)

costs = np.sum((f(x.reshape((1, N)), a_ests.reshape((M, 1))) - y)**2, axis=1)
plt.figure("Cost wrt. a")
plt.scatter(a_ests, costs)