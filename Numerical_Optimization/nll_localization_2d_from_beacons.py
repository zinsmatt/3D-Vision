from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import maximum_fill_value

# np.random.seed(7)

N = 4
W = 100
H = 100
# beacons = np.random.rand(N, 2)
# beacons[:, 0] *= W
# beacons[:, 1] *= H
beacons = np.array([[75.0, 65.0],
                    [77.0, 45.0],
                    [60.0, 70.0],
                    [40.0, 60.0]])
print("========================")
print("======= beacons ========")
print("========================")
print(beacons)
print("========================")

def f(x):
    return np.sqrt(np.sum((beacons - x)**2, axis=1))

def fi(xs, i):
    return np.sqrt(np.sum((xs - beacons[i, :])**2, axis=1))

def Je(x):
    ff = f(x)
    d = x - beacons
    if np.min(ff) < 1e-5:
        print("Warning sqrt not differentiable in 0 !!!")
        ff[ff < 1e-5] += 0.1
    d[:, 0] /= ff
    d[:, 1] /= ff
    return d



# Ground-truth position
X = np.random.rand(2)
X[0] *= W
X[1] *= H
# X = np.array([30.0, 27.0])
measure = f(X)
print("GT loc: ", measure)

# Display cost function
yy, xx = np.mgrid[:H, :W]
pts = np.vstack((xx.flatten(), yy.flatten())).T
dists = np.zeros((H, W), dtype=float)
for i in range(N):
    dists += (fi(pts, i).reshape((H, W)) - measure[i])**2



plt.contourf(xx, yy, dists, levels=100)
plt.scatter(beacons[:, 0], beacons[:, 1], color="red")
plt.scatter(X[0], X[1], color="green")


x = np.array([70.0, 30.0])
plt.scatter(x[0], x[1], color="blue")

# plt.show()

J = Je(x)
print("J = ", J)

MAX_ITER = 10000
cost_threshold = 0.01
step_size = 1
e = f(x) - measure
cost_prev = 999999999
lm = 0.1
viz = [x.copy()]
for i in range(MAX_ITER):
    x_prev = x.copy()

    J = Je(x) # Jacobian of e(x)

    # step_size = 0.1   # for Gradient descent
    # delta = -J.T.dot(e)   # Gradient descent

    # delta = -(np.linalg.inv(J.T @ J) @ J.T).dot(e)   # Gauss-Newton

    delta = -(np.linalg.inv(J.T @ J + (np.eye(2)*lm)) @ J.T).dot(e)  # Levenberg-Marquardt


    x += delta * step_size
    e = f(x) - measure
    cost = np.sum(e**2)

    # Update the factor term of LM
    if cost > cost_prev:
        # Cost increased => increase the factor to go towards a small gradient descent step
        x = x_prev
        lm *= 1.5
        continue
    else:
        # Cost decreased => reduce the factor to act more like a Newton method
        lm *= 0.8

    viz.append(x.copy())
    if cost < cost_threshold:
        break
    if np.abs(cost - cost_prev) < 0.01:
        break
    cost_prev = cost

print(len(viz), "iterations")
viz = np.vstack(viz)
plt.plot(viz[:, 0], viz[:, 1], "-", color="blue")
plt.show()
print("Final error: ", np.linalg.norm(x-X))