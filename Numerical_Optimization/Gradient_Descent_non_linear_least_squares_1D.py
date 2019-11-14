#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Residual and its first derivative
def r(x):
    return 1-np.exp(-0.1*x)

def grad_r(x):
    return np.exp(-x)

# Total "cost" function
def F(x):
    return 0.5 * r(x)**2


# parameters
eps = 1e-12
xmin, xmax = -10, 10
x0 = -10
MAX_ITER = 200
it = 0


# plot the curve
X = np.linspace(xmin, xmax, 1000)
Y = F(X)
plt.plot(X, Y)


# Here the total objective function that is minimized is called "cost"
# and the 1d residual is r(). (In Ceres a costFunction corresponds to a 
# residual block)
cost = F(x0)
prev_cost = cost + 1

x = x0
values = [x0]   # keep the positions for visualization

while abs(prev_cost - cost) > eps and it < MAX_ITER:
    prev_cost = cost
    
    # steepest descent direction is -F'(x)
    h = -grad_r(x) * r(x)
    if np.linalg.norm(h) < eps:
        break
    h /= np.linalg.norm(h)
    
       
    # line search phi(a) = F(x + ah)
    # => we want to find "a" for which phi(a) is minimum 
    # => iterative search until |phi'(a)| <= lambda * |phi'(0)|
    # => in this implementation we take the minimum absolute derivative
    line_search_radius = 5
    a = np.linspace(0, line_search_radius, 50)
    x_local = x + a*h
    phi_deriv = h * (grad_r(x_local) * r(x_local))
    min_a_idx = np.argmin(np.abs(phi_deriv))
    # force alpha at least 0.01 => we only stop if the derivative h is 0
    # if alpha is remains very small, the line_search_radius could be increased
    alpha = max(0.01, a[min_a_idx])
    
    # update x
    x = x + alpha * h
    
    cost = F(x)
    it += 1
    print("Iteration %d: %.4f x = %.4f" % (it, cost, x))
    values.append(x)

values = np.vstack(values)
plt.plot(values[:,0], 0.5*r(values[:, 0])**2, marker='+', markersize=10, lineWidth=1, label="Gradient Descent")    
plt.legend()
print("Gradient Descent converged in %d iterations" % it)
