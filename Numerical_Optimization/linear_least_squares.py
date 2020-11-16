import numpy as np
import matplotlib.pyplot as plt

""" === Linear Least-Squares ===
The model is linear.
=> The cost function is create from squared distance
=> which gives a quadratic cost function. (of the for f(x) = x.T Q x + L x + c)
=> It is possible to find the minimum of the cost function analytically.
=> This minimum is the global minimum.
"""


a = 1.35
b = 2.33
N = 50
noise = np.random.randn(N)
x = np.linspace(0, 10, N)
y = a * x + b + noise



plt.scatter(x, y)
plt.ylim([0, 2 + np.max(y)])
plt.xlim([0, 10])

M = 50
a_ests = np.linspace(-5, 5, M)
b_ests = np.linspace(-35, 35, M)


# cost function wrt. a
costs = []
for a_est in a_ests:
    y_est = a_est * x + b
    cost = np.sum((y_est - y)**2)
    costs.append(cost)


# cost function wrt. a
costs_b = []
for b_est in b_ests:
    y_est = a * x + b_est
    cost_b = np.sum((y_est - y)**2)
    costs_b.append(cost_b)

# (developed cost function)
costs2 = []
for a_est in a_ests:
    cost2 = np.sum(a_est**2 * x**2 + (b - y)**2 + 2 * a_est * x * (b-y))
    costs2.append(cost2)


a_est = 1.0 / x.dot(x) * x.dot(y - b)


plt.plot([-10, 10], [-10*a_est+b, 10*a_est+b], c="green")


plt.figure("cost wrt. a")
plt.scatter(a_ests, costs, c="blue")
# plt.scatter(a_ests, costs2, c="red")
plt.plot([a_est, a_est], [-10, 14000])

plt.figure("cost wrt. b")
plt.scatter(b_ests, costs_b, c="blue")



a_ests, b_ests = np.meshgrid(a_ests, b_ests)
costs_2d = np.sum((a_ests.reshape((M, M, 1)) * x.reshape((1, N)) + b_ests.reshape((M, M, 1)) - y.reshape((1, N)))**2, axis=2)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(a_ests, b_ests, costs_2d, cmap=("jet"))
plt.figure("cost2d")
plt.imshow(costs_2d)
