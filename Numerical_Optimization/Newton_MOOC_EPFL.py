#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:10:10 2021

@author: mzins
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return x**2-2
    # return x - math.sin(x)
    # return math.atan(x)

def f_prime(x):
    return 2*x
    # return 1-math.cos(x)
    # return 1.0 / (1.0 + x**2)


def Newton(x_init, thresh=1e-15): 
    f_cur = f(x_init)
    x = x_init
    while abs(f_cur) > thresh:
        plt.scatter(x, f_cur, c="red")
        print(x)
        deriv = f_prime(x)
        if deriv == 0.0:
            print("Error div by 0")
            return -1, -1
        x -= f_cur / deriv
        f_cur = f(x)
    plt.scatter(x, f_cur, c="green")
    return x, f_cur


plt.plot([-2.5, 2.5], [0.0, 0.0])

x_range = np.linspace(-2.5, 2.5, 100)
y = [f(x) for x in x_range]
plt.plot(x_range, y)

x_init = 2
# x_init = 1
# x_init = 1.5

x_star, value = Newton(x_init)
print("f at %.4f = %.4f" %  (x_star, value))



#%%
# import numpy as np
# import math

# def f(x):
#     f0 = x[0]**3-3*x[0]*x[1]**2-1
#     f1 = x[1]**3-3*x[0]**2*x[1]
#     return np.array([f0, f1])
    

# def f_prime(x):
#     J = np.array([[3*x[0]**2-3*x[1]**2, -6*x[0]*x[1]],
#                   [-6*x[0]*x[1], 3*x[1]**2-3*x[0]**2]])
#     return J


# def Newton(x_init, thresh=1e-15): 
#     f_cur = f(x_init)
#     x = x_init
#     while np.linalg.norm(f_cur) > thresh:
#         print(x)
#         J = f_prime(x)
#         if np.linalg.det(J) == 0.0:
#             print("Error Jacobian not invertible")
#             return -1, -1
    
#         J_inv = np.linalg.inv(J)
#         x -= J_inv @ f_cur
#         f_cur = f(x)
#     return x, f_cur


# x_init = np.array([-2.0, -2.0])


# x_star, value = Newton(x_init)
# print("res at ");
# print(x_star)
# print("value = ")
# print(value)


#%%

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

def f(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return 2*x1**3 + 6*x1*x2**2 - 3*x2**3 - 150*x1

    
def grad(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return np.array([[6*x1**2+6*x2**2-150],
                     [12*x1*x2-9*x2**2]], dtype=float)
    
def Hessian(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return np.array([[12*x1, 12*x2],
                     [12*x1, 12*x1-18*x2]], dtype=float)


def Newton_optim(x_init, thresh=1e-8):
    cost = f(x_init)
    x = x_init
    g = grad(x)
    
    path = [x_init.copy()]
    costs = [cost]
    while np.linalg.norm(g) > thresh:
        print("x = ", x)

        H = Hessian(x)
        print(H)
        print(g)
        d = -np.linalg.inv(H) @ g
        x += d
        
        cost = f(x)
        g = grad(x)
        path.append(x.copy())
        costs.append(cost)
    path = np.hstack(path).T
    return x, g, cost, path, costs
        


# plt.plot([-10, 10], [0.0, 0.0])
# x_range = np.linspace(-10, 10, 100)
# y = [f(x) for x in x_range]
# plt.plot(x_range, y)
N = 40
dx = np.linspace(-8.0, 8.0, N)
dy = np.linspace(-8.0, 8.0, N)
y, x = np.meshgrid(dx, dy)
z = []
for xx, yy in zip(x.flatten(), y.flatten()):
    v = f(np.array([xx, yy]).reshape(2, 1))
    z.append(v)
z = np.array(z).reshape((N, N))


fig = plt.figure()
ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)


# plt.contourf(x, y, z, 128)
ax.contour(x, y, z, 64)

# x_init = np.array([6.2, -3.4]).reshape((2, 1))
x_init = np.array([4.2, -6.4]).reshape((2, 1))
# x_init = np.array([2.8, 4.3]).reshape((2, 1))


x_star, grad, value, path, zs = Newton_optim(x_init)
print("res at ");
print(x_star)
print("value ", value)
print("grad norm = ", np.linalg.norm(grad))


ax.plot(path[:, 0], path[:, 1], np.array(zs), 'r--+')
# ax.plot(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

