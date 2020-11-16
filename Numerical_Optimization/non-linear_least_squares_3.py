import numpy as np
import matplotlib.pyplot as plt


""" === Non-linear Least-squares ===
Strange example where our model is linear (we want to fit a linear function),
but non-linear wrt. the parameter we want to estimate.
=> The cost function is not a quadratic function anymore.
=> The cost function has local minima.
=> No closed-form solution exists.
"""

def f(x, a):
    return a**2 *x + a**3

a = 1.35
N = 100
x = np.linspace(-1, 1, N)
y = f(x, a)

plt.scatter(x, y, s=1)


M = 1000
a_ests = np.linspace(-2, 2, M)

costs = np.sum((f(x.reshape((1, N)), a_ests.reshape((M, 1))) - y)**2, axis=1)
plt.figure("Cost wrt. a")
plt.scatter(a_ests, costs, s=1)