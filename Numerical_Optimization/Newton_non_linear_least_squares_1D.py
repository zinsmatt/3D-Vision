#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def f(x):
    return 1-np.exp(-x)

def grad_f(x):
    return np.exp(-x)

def grad2_f(x):
    return -np.exp(-x)


eps = 1e-12
xmin, xmax = -1.5, 4.0

# Residual and its first and second derivatives
r = f
grad_r = grad_f
grad2_r = grad2_f

X = np.linspace(xmin, xmax, 1000)
Y = 0.5 * r(X)**2
plt.plot(X, Y)

x0 = 0.5

# Here the total objective function that is minimized is called "cost"
# and the 1d residual is r(). (In Ceres a costFunction corresponds to a 
# residual block)
cost = 0.5 * r(x0)**2
prev_cost = cost + 1

MAX_ITER = 200
it = 0

x = x0
values = [x0]
while abs(prev_cost - cost) > eps and it < MAX_ITER:
    prev_cost = cost
    
    deriv = grad_r(x) * r(x)
    deriv2 = grad_r(x) * grad_r(x) + grad2_r(x) * r(x)
    # in Gauss-Newton the second part of this  expression is neglected because 
    # r(x) should be small (this avoids computing the second derivative grad2_r)
    
    
    # second-order approximation (quadratic function)
    xx = np.linspace(x-0.5, x+0.5, 100)
    approx_quadratic = 0.5*r(x)**2 + deriv * (xx - x) + 0.5 * deriv2 * (xx - x)**2
    plt.plot(xx, approx_quadratic.flatten())
    
    # we are looking for the min of this approximation
    #    f'(x + dx) = 0 = f'(x) + f''(x) * dx
    # => dx = -f'(x) / f''(x)
    #
    # WARNING !!! it is important that f''(x) > 0
    # otherwise we are going to find a maximum of the approximaiton
    # (for N-dimensional case the Hessian matrix should be positive
    # definite)
    # => this is probably related to the condition that x0 should be 
    # relatively "close" to the solution
    #
    
    if deriv2 < 0:
        print("WARNING: negative second derivative => diverge")
        break
    
        
    d = -deriv / deriv2
    x = x + d
    
    cost = 0.5 * r(x)**2
    it += 1
    print("Iteration %d: %.4f x = %.4f" % (it, cost, x))
    values.append(x)

values = np.vstack(values)
plt.plot(values[:,0], 0.5*r(values[:, 0])**2, marker='+', markersize=10, lineWidth=1, label="Newton")    

print("Newton: Converged towards : %.2f in %d iterations" % (x, it))
plt.legend()