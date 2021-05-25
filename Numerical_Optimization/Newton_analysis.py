#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:04:01 2021

@author: mzins
"""

import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return -x**4 + 12 * x**3 - 47 * x**2 + 60 * x

def grad(x):
    return -4*x**3+36*x**2-94*x+60

def grad2(x):
    return -12*x**2+72*x-94

def model(x, xc):
    return f(xc) + (x - xc) * grad(xc) + 0.5 * (x - xc)**2 * grad2(xc)

def zero(x):
    return x - grad(x) / grad2(x)


N = 50
x_range = np.linspace(0.0, 5, N)

y = [f(x) for x in x_range]

plt.plot(x_range, y)

xc = 3.0
y_model = model(x_range, xc)

plt.plot(x_range, y_model, "green")

plt.ylim([-10, 26])

z = zero(xc)

plt.scatter(xc, model(xc, xc), s=14, c="orange")
plt.scatter(z, model(z, xc), s=14, c="red")