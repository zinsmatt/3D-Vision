#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:31:37 2021

@author: mzins
"""

import numpy as np
import matplotlib.pyplot as plt



def f(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return x2**2+11*x2**2+x1*x2


def grad(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return np.array([[2*x1+x2],
                     [22*x2+x1]])

def grad2(x):
    return np.array([[2.0, 1.0],
                     [1.0, 22.0]])

def optimize(x_init, D, thresh):
    cost = f(x_init)
    x = x_init
    g = grad(x)
    H = grad2(x)
    
    while np.linalg.norm(g) > thresh:
        dk = -D @ g
        
        alpha = -dk.T.dot(g) / (dk.T.dot(H).dot(dk))
        print("===> ", alpha)
        x = x + alpha * dk
        
        cost = f(x_init)
        g = grad(x)
        H = grad2(x)
        print(np.linalg.norm(g))

    return x, cost, g
       
        
x_init = np.array([[4.0],
                   [1.0]])
D = np.eye(2, dtype=float)

x_star, f_star, grad_star = optimize(x_init, D, 1e-7)

print("min at ", x_star.flatten())
print("f_star = ", f_star.flatten())
print("grad = ", grad_star.flatten())
