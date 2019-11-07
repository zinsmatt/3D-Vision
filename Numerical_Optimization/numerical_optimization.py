#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import matplotlib.pyplot as plt



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

#%%
# visualization
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f_rosen(X, Y)
plt.imshow(Z, interpolation="bicubic", 
           origin="lower",
           cmap="nipy_spectral",
           extent=[-2, 2, -2, 2])
    
#%%
# GD
init_range = 4.0
x0 = (np.random.random(2) - 0.5) * init_range 

eps = 1e-5

cost = f_rosen(*x0)
loss = cost
prev_loss = loss + 1

x = x0
it = 0
step = 0.01

values = [x0]
while prev_loss - loss > eps:
    prev_loss = loss
    
    grad = grad_f_rosen(*x)
    grad /= np.linalg.norm(grad)
    x = x - step * grad
    it += 1
    
    values.append(x)
    
    loss = f_rosen(*x)
    print("Loss at iter %d : %.4f" % (it, loss))
        
values = np.vstack(values)
plt.plot(values[:,0], values[:, 1], marker='+', markersize=2, lineWidth=1)    


