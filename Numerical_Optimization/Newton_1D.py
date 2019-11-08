#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



def f_convex1d(x):
    return 2.0-np.exp(-(x - 1.0)**2)
        
def grad_f_convex1d(x):
    return 2*(x-1)*np.exp(-(x-1)**2)

def grad2_f_convex1d(x):
    return 2*(-2*(x-1)*np.exp(-(x-1)**2) + np.exp(-(x-1)**2))


def f_convex1d_2(x):
    return (x-1)**2
        
def grad_f_convex1d_2(x):
    return 2*(x-1)

def grad2_f_convex1d_2(x):
    return 2*x


def f_nonconvex1d_1(x):
    return x**4 - 14*x**3 + 60*x**2 - 70*x
        
def grad_f_nonconvex1d_1(x):
    return 4*x**3 - 14*3*x**2 + 2*60*x - 70

def grad2_f_nonconvex1d_1(x):
    return 4*3*x**2 - 14*3*2*x + 2*60

def f_test(x):
    return (1-np.exp(-x))**2
def grad_f_test(x):
    return 2*(1-np.exp(-x))*2*np.exp(-x)
def grad2_f_test(x):
    return 2*np.exp(-2*x)-2*(1-np.exp(-x))*np.exp(-x)


eps = 1e-12
xmin, xmax = -0.5, 4.0


F = f_nonconvex1d_1
grad_F = grad_f_nonconvex1d_1
grad2_F = grad2_f_nonconvex1d_1

X = np.linspace(xmin, xmax, 1000)
Y = F(X)
plt.plot(X, Y)

x0 = -0.33

cost = F(x0)
prev_cost = cost + 1

MAX_ITER = 200
it = 0

x = x0
values = [x0]
while abs(prev_cost - cost) > eps and it < MAX_ITER:
    prev_cost = cost
    
    deriv = grad_F(x)
    deriv2 = grad2_F(x)
    
    # second-order approximation
    xx = np.linspace(x-0.5, x+0.5, 100)
    approx_quadratic = F(x) + deriv * (xx - x) + 0.5 * deriv2 * (xx - x)**2
    plt.plot(xx, approx_quadratic.flatten())
    
    # we are looking for the min of this approximation
    #    f'(x + dx) = 0 = f'(x) + f''(x) * dx
    # => dx = -f'(x) / f''(x)
    #
    # WARNING !!! it is important that f''(x) > 0
    # otherwise we are going to find a maximum of the approximaiton
    # (for N-dimensional case the Hessian matrix should be positive
    # definite)
    #
    
    if deriv2 < 0:
        print("WARNING: negative second derivative => diverge")
        break
    
        
    d = -deriv / deriv2
    x = x + d
    
    cost = F(x)
    it += 1
    print("Iteration %d: %.4f" % (it, cost))
    values.append(x)

values = np.vstack(values)
plt.plot(values[:,0], F(values[:, 0]), marker='+', markersize=10, lineWidth=1, label="Newton")    

print("Newton: Converged towards : %.2f in %d iterations" % (x, it))
plt.legend()