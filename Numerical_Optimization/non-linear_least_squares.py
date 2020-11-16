import numpy as np
import matplotlib.pyplot as plt


""" === Non-linear Least-squares ===
We want to fit a non-linear function which is not linear wrt. the parameter
we want to estimate.
=> The cost function is not a quadratic function anymore.
=> The cost function has local minima.
=> No closed-form solution exists.


=> solve with exhaustive search

=> solve with gradient descent. Just compute the gradient

=> solve with second-order method (Newton).
=> compute the gradient + second order derivate (hessian).
=> compute the second-order approximation of the cost function.
=> the second-order approximation is a quadratic function.
=> Find the minimum and iterate
"""

def f(x, a):
    return a**2 * x**3 - a * x**2 + x

a = 1.35
N = 100
noise = np.random.randn(N) * 0.1
x = np.linspace(-1, 1, N)
y = f(x, a) + noise

plt.scatter(x, y, s=1)


M = 1000
a_ests = np.linspace(-2, 2, M)

costs = 0.5 * np.sum((f(x.reshape((1, N)), a_ests.reshape((M, 1))) - y.reshape((1, N)))**2, axis=1)
plt.figure("Cost wrt. a")
plt.scatter(a_ests, costs, s=1)
plt.xlim([-2, 2])


#%% Exhaustive search (not efficient)
# best_index = np.argmin(costs)
# a_est = a_ests[best_index]

# print("gta = ", a)
# print("estimate a = ", a_est)

# plt.figure("Estimated curve")
# x2 = np.linspace(-1, 1, 10*N)
# plt.scatter(x, y, s=1)
# plt.scatter(x2, f(x2, a_est), c="green", s=1)


#%% Gradient Descent

def cost(x, a, y):
    return 0.5 * np.sum((f(x, a) - y)**2)

def grad_cost(x, a, y):
    return np.sum((2*a*x**3-x**2)*(f(x, a) - y))



plt.figure("Gradient Descend")

costs = np.array([cost(x, a_est, y) for a_est in a_ests])
plt.scatter(a_ests, costs, s=1)


a0 = -1.9
step = 0.001
MAX_ITER = 100000
size_tgt = 0.5

a_cur = a0
a_values = [a0]
cost_values = [cost(x, a0, y)]
for i in range(MAX_ITER):
    g = grad_cost(x, a_cur, y)

    pts = np.array([[a_cur + size_tgt, cost(x, a_cur, y) + size_tgt * g],
                    [a_cur - size_tgt, cost(x, a_cur, y) - size_tgt * g]])
    plt.plot(pts[:, 0], pts[:, 1], "g--")

    a_cur = a_cur - g * step
    a_values.append(a_cur)
    cost_values.append(cost(x, a_cur, y))
    if abs(g) < 1e-1:
        break

plt.scatter(a_values, cost_values)
print("Stopped after %d iter" % i)
print("estimated a = %.4f" % a_cur)

plt.figure("Estimated parameter")
plt.scatter(x, y, s=1)
plt.scatter(x, f(x, a_cur), s=1, c="red")



#%%

def deriv_second_cost(x, a, y):
    return np.sum(2*x**3*(a**2*x**3-a*x**2+x-y)+(2*a*x**3-x**2)*(2*a*x**3-x**2))


plt.figure("Newton Method")
costs = np.array([cost(x, a_est, y) for a_est in a_ests])
plt.scatter(a_ests, costs)

a0 = -1.5
MAX_ITER = 1000
size_approx = 0.3


a_cur = a0
a_values = [a0]
cost_values = [cost(x, a0, y)]
for i in range(MAX_ITER):
    c0 = cost(x, a_cur, y)
    g = grad_cost(x, a_cur, y)
    g2 = deriv_second_cost(x, a_cur, y)
    if abs(g) < 1e-1:
        break

    aa = np.linspace(-size_approx, size_approx, 100)
    values = c0 + g * aa + 0.5 * g2 * aa**2

    a_best = (g2 * a_cur - g) / g2 # best a

    a_values.append(a_best)
    cost_values.append(cost(x, a_best, y))

    plt.scatter(a_cur + aa, values, c='red', s=1)
    v = c0 + g * (a_best - a_cur) + 0.5 * g2 * (a_best - a_cur)**2
    plt.plot([a_best, a_best], [cost_values[-1] - 20, cost_values[-1] + 20])
    plt.ylim([-1, 200])
    plt.xlim([-3, 3])

    a_cur = a_best
plt.scatter(a_values, cost_values)
print("Stopped after %d iter" % i)
print("Estimate parameter = %.4f" % a_cur)
plt.figure("Estimated parameter")
plt.scatter(x, y, s=1)
plt.scatter(x, f(x, a_cur), s=1, c="red")



