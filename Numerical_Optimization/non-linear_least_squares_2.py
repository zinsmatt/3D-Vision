import numpy as np
import matplotlib.pyplot as plt


""" === Non-linear Least-squares ===
Estimate the 2D rigid transform between two sets of points
=> The cost function is not a quadratic function.
=> The cost function has local minima.
=> No closed-form solution exists.
"""

def rot_mat(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def f(X, angle, t):
    return (rot_mat(angle) @ X.T + t.reshape((-1, 1))).T

X = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [1.0, 1.0],
              [0.0, 1.0]])

angle = np.deg2rad(55);
R = rot_mat(angle)

t = np.array([2.34, 1.43])

noise = np.random.randn(4, 2) * 0.1
Y = f(X, angle, t) + noise


plt.scatter(X[:, 0], X[:, 1], c="red")
plt.scatter(Y[:, 0], Y[:, 1], c="blue")
plt.xlim([-2, 8])
plt.ylim([-2, 8])
plt.gca().set_aspect(1)


M = 100
a_ests = np.linspace(-np.pi, np.pi, M)
costs = []
for a_est in a_ests:
    cost = np.sum((f(X, a_est, t) - Y)**2)
    costs.append(cost)


plt.figure("Cost wrt. angle")
plt.scatter(a_ests, costs, s=1)

plt.figure("Cost wrt. angle and t[0]")
t0_ests = np.linspace(0, 10, M)
a_ests_2d, t0_ests_2d = np.meshgrid(a_ests, t0_ests)

costs_2d = []
for a_est, t0_est in np.vstack((a_ests_2d.flatten(), t0_ests_2d.flatten())).T:
    cost = np.sum((f(X, a_est, np.array([t0_est, t[1]])) - Y)**2)
    costs_2d.append(cost)

costs_2d = np.asarray(costs_2d).reshape((M, M))

plt.imshow(costs_2d, cmap="jet")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(a_ests_2d, t0_ests_2d, costs_2d, cmap=("jet"))

#%% Estimation by exhaustive search (not efficient)
best_index = np.argmin(costs_2d.flatten())
a_est = a_ests_2d.flatten()[best_index]
t_est = np.array([t0_ests_2d.flatten()[best_index], t[1]])
Y_filt = f(X, a_est, t_est)

plt.figure("Estimation")
plt.scatter(X[:, 0], X[:, 1], c="red")
plt.scatter(Y[:, 0], Y[:, 1], c="blue")
plt.scatter(Y_filt[:, 0], Y_filt[:, 1], c="green")
plt.xlim([-2, 8])
plt.ylim([-2, 8])
plt.gca().set_aspect(1)