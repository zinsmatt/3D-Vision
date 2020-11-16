import numpy as np
import matplotlib.pyplot as plt


""" === Non-linear Least-squares ===
We want to fit a non-linear function which is not linear wrt. the parameter
we want to estimate.
=> The cost function is not a quadratic function anymore.
=> The cost function has local minima.
=> No closed-form solution exists.
"""

def f(x, a):
    return a**2 * x**3 - a * x**2 + x

a = 1.35
N = 100
noise = np.random.randn(N) * 0.3
x = np.linspace(-1, 1, N)
y = f(x, a) + noise

plt.scatter(x, y, s=1)


M = 1000
a_ests = np.linspace(-2, 2, M)

costs = np.sum((f(x.reshape((1, N)), a_ests.reshape((M, 1))) - y)**2, axis=1)
plt.figure("Cost wrt. a")
plt.scatter(a_ests, costs, s=1)


#%% solve by exhaustive search (not efficient)
best_index = np.argmin(costs)
a_est = a_ests[best_index]

print("gta = ", a)
print("estimate a = ", a_est)

plt.figure("Estimated curve")
x2 = np.linspace(-1, 1, 10*N)
plt.scatter(x, y, s=1)
plt.scatter(x2, f(x2, a_est), c="green", s=1)