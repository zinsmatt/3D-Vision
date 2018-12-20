#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np


def skew(x):
    """ return the skew matrix for cross product """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def linear_triangulate(p1, p2, M1, M2):
    """ Estimate 3D locations of points from two observations
        in two images by triangulation """
    pts_est = []
    for i in range(p1.shape[1]):
        # prepare linear system
        A1 = skew(p1[:, i]) @ M1
        A2 = skew(p2[:, i]) @ M2
        A = np.vstack((A1, A2))
        # solve linear system
        U, S, Vt = np.linalg.svd(A)
        res = Vt[-1, :]
        pts_est.append(res)

    pts_est = np.vstack(pts_est).T
    # dehomogeneize
    pts_est /= pts_est[-1, :]
    return pts_est[:3, :]


M1 = np.array([[500, 0, 320, 0],
               [0, 500, 240, 0],
               [0, 0, 1, 0]])

M2 = np.array([[500, 0, 320, -100],
               [0, 500, 240, 0],
               [0, 0, 1, 0]])

N = 15
# random 3D points
pts = np.random.rand(4, N)
pts[2, :] = pts[2, :] * 5 + 10
pts[3, :] = 1

# points observations in im1 and im2
p1 = M1 @ pts
p2 = M2 @ pts

# triangulate
pts_est = linear_triangulate(p1, p2, M1, M2)

# compute triangulation residuals
residuals = np.sqrt(np.sum((pts_est - pts[:3, :])**2, axis=0))
print("Linear Triangulation residuals : ", residuals)
