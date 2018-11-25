#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import matplotlib.pyplot as plt

NB = 20

# create points lying on a line with noise
points = np.c_[np.linspace(0, 10, NB), np.linspace(0, 30, NB)]
sigma = 1.0
noise = np.random.rand(NB, 2) * sigma
points += noise


# estimate the "best" line
# turn points to homogeneous coordinates
points = np.c_[points,  np.ones((NB, 1))]

# If all points were lying on a single line l, it would verify:
#      points . l = 0   (this is not the case in reality because of noise)
# To estimate the "best" line, we minimize ||points.l|| using SVD
U, S, Vt = np.linalg.svd(points)

# l_est is the dual homogeneous coordinates of the 2D line
l_est = Vt[2, :]
print("dual homogeneous coordinates of l = ", l_est)

# get the canonical estimate (D-normalization)
l_est_n = (-np.sign(l_est[2]) / np.linalg.norm(l_est[:2])) * l_est
print("canonical representation of l = ", l_est_n)


residuals = np.abs(points @ l_est_n)
print("Mean error = ", np.mean(residuals))
print("Max error = ", np.max(residuals))

# plot points
plt.scatter(points[:, 0], points[:, 1])

# plot the line
p0 = [0, -l_est_n[2] / l_est_n[1]]
p1 = [10, (-l_est_n[2] - l_est_n[0] * 10) / l_est_n[1]]
line = np.vstack((p0, p1))
plt.plot(line[:, 0], line[:, 1], "r")

# ***** methode 2 (better) *****
# substract the centroid
c = np.mean(points[:, :2], axis=0)
points_2 = points[:, :2] - c

U2, S2, Vt2 = np.linalg.svd(points_2)

t = Vt2[0, :]
t /= np.linalg.norm(t)

p0 = c - 10 * t
p1 = c + 10 * t
line2 = np.vstack((p0, p1))
plt.plot(line2[:, 0], line2[:, 1], "g")
