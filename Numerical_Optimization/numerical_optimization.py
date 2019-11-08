#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# =============================================================================
# This file contains some iterative algorithm for unconstrained non-linear
# minimization. For example, the Rosenbrock's function is tried to be minimized.
# =============================================================================

def f_rosen(x, y):
    """
        Cost function that we want to optimize.
        (This is Rosenbrock's function and 
        have a global minimum in (1, 1))
    """
    return (1.0 - x)**2 + 100 * (y - x**2)**2

def grad_f_rosen(x, y):
    """
        Return the gradient of Rosenbrock's function
        in (x, y)
    """
    return np.array([400*x**3-400*y*x+2*x-2, 
                     -200*x**2+200*y])


F = f_rosen
grad_F = grad_f_rosen

#%%
# visualization
xmin, xmax = -5.0, 5.0
ymin, ymax = -5.0, 5.0
x = np.linspace(xmin, xmax, 100)
y = np.linspace(ymin, ymax, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * F(X, Y)**2
plt.imshow(Z, interpolation="bicubic", 
           origin="lower",
           cmap="hsv",
#           norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()*0.001),
           norm=colors.PowerNorm(0.1),
           extent=[xmin, xmax, ymin, ymax])

    
#%% Gradient Descent

init_range = 10.0
x0 = (np.random.random(2) - 0.5) * init_range 
x0 = np.array([-4.0, 2.0])

eps = 1e-12

cost = F(*x0)
loss = 0.5 * cost**2
prev_loss = loss + 1

x = x0
MAX_ITER = 5000
it = 0
step = 0.05
use_linear_search_along_gradient = True

if use_linear_search_along_gradient:
    possible_steps = np.linspace(0.0001, 0.1, 10000).reshape((-1, 1))
values = [x0]
while abs(prev_loss - loss) > eps and it < MAX_ITER and np.linalg.norm(x - [1.0, 1.0]) > 1e-4:
    prev_loss = loss
    
    grad = grad_F(*x) * F(*x)
    grad /= np.linalg.norm(grad)
    
    if use_linear_search_along_gradient:
        temp = -((possible_steps * grad) - x)
        best_idx = np.argmin(F(temp[:, 0], temp[:, 1]))
        best_step = possible_steps[best_idx]
        step = best_step
    
    x = x - step * grad
    loss = 0.5 * F(*x)**2
    
    if not use_linear_search_along_gradient:
        # Heuristic-based step update
        if loss > prev_loss:
            step = max(0.0001, step / 2)
        else:
            step = min(0.1, step * 1.5)
    
    it += 1
    values.append(x)
#    print("Loss at iter %d : %.8f" % (it, loss))
        
values = np.vstack(values)
plt.plot(values[:,0], values[:, 1], marker='+', markersize=2, lineWidth=1, label="Gradient Descent")    

print("Gradient Descent: Converged towards : %.2f %.2f in %d iterations" % (x[0], x[1], it))



#%% Conjugate Gradients

#init_range = 10.0
#x0 = (np.random.random(2) - 0.5) * init_range 

cost = F(*x0)
loss = 0.5 * cost**2
prev_loss = loss + 1

MAX_ITER = 5000
it = 0

step = 0.001

x = x0
g =  grad_F(*x) * F(*x)
g /= np.linalg.norm(g)
h = g
values = [x0]
possible_steps = np.linspace(0.0001, 0.1, 10000).reshape((-1, 1))
while abs(prev_loss - loss ) > eps and it < MAX_ITER and np.linalg.norm(x - [1.0, 1.0]) > 1e-4:
    prev_loss = loss
    g_prev = g
    h_prev = h
    
    # find the best step
    temp = -(possible_steps * h - x)
    best_idx = np.argmin(F(temp[:, 0], temp[:, 1]))
    best_step = possible_steps[best_idx]
    
    x = x - best_step * h
    loss = 0.5 * F(*x)**2
    
    g = grad_F(*x) * F(*x)
    g /= np.linalg.norm(g)
    h = g + h_prev * (g - g_prev).dot(g) / (g_prev.dot(g_prev))
    
    it += 1
#    print("Loss at iter %d : %.8f" % (it, loss))
    
    values.append(x)

 
values = np.vstack(values)
plt.plot(values[:,0], values[:, 1], marker='+', markersize=2, lineWidth=1, label="Conjugate Gradient")    

print("Conjugate Gradient: Converged towards : %.2f %.2f in %d iterations" % (x[0], x[1], it))

plt.legend()

#%% Gauss-Newton

def f_convex1d(x):
    return -np.exp(-(x - 1.0)**2)
            #.0 / ((x -3)**2+0.1) + 1.0 / ((x-2)**2+0.05) + 2.0
        
def grad_f_convex1d(x):
#    return -0.4*(x-0.7)**2 * -np.exp(-0.4*(x - 0.7)**2) * -0.4 * 2 * (x - 0.7)
    return -2.0*(x-1.0)*np.exp((x-1)**2)

F = f_convex1d
grad_F = grad_f_convex1d

X = np.linspace(xmin, xmax, 1000)
Y = F(X)
plt.plot(X, Y)

x0 = np.array([2.5])

cost = F(*x0)
loss = 0.5 * cost**2
prev_loss = loss + 1

MAX_ITER = 100
it = 0

step = 0.001

x = x0
values = [x0]
possible_steps = np.linspace(0.0001, 0.1, 10000).reshape((-1, 1))
while abs(prev_loss - loss ) > eps and it < MAX_ITER and abs(x-1.0) > 1e-4:
    prev_loss = loss
    
    J = grad_F(*x).reshape((1, -1))
    if (abs(J[0]) < 1e-9):
        break
    d = -np.linalg.inv(J.dot(J.T)) * J * F(*x)
    d = d.flatten()
    print(J, d)
    
    x = x + d
    loss = 0.5 * F(*x)**2
    it += 1
    
    values.append(x)

values = np.vstack(values)
#plt.plot(values[:,0], values[:, 1], marker='+', markersize=2, lineWidth=1, label="Gauss Newton")    
plt.plot(values[:,0], F(values), marker='+', markersize=2, lineWidth=1, label="Gauss Newton")    

#print("Gauss Newton: Converged towards : %.2f %.2f in %d iterations" % (x[0], x[1], it))
print("Gauss Newton: Converged towards : %.2f in %d iterations" % (x[0], it))

plt.legend()