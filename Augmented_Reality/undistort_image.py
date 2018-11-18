#!/usr/bin/env python3
"""
@author: matt
"""


import cv2
import numpy as np
import scipy.interpolate

image = cv2.imread("data/images/img_0001.jpg", cv2.IMREAD_GRAYSCALE)
k1, k2 = np.loadtxt("data/D.txt")
K = np.loadtxt("data/K.txt")


y, x = np.mgrid[:image.shape[0], :image.shape[1]]
pts = np.c_[x.flatten(), y.flatten(),
            np.ones((image.shape[0] * image.shape[1]))]

# Normalized points
pts_norm = np.linalg.inv(K).dot(pts.T)


r = pts_norm[0, :]**2 + pts_norm[1, :]**2

pts_norm_dist = pts_norm[:2, :] * (1 + k1 * r + k2 * r**2)
pts_norm_dist = np.r_[pts_norm_dist, np.ones((1, pts_norm_dist.shape[1]))]

pts_dist = K.dot(pts_norm_dist)

pts_dist = pts_dist[:2, :].T

# Nearest-Neighbour interpolation
# pts_dist = np.round(pts_dist[:2, :]).astype(int).T
# pixels = image[pts_dist[:, 1], pts_dist[:, 0]]

# Linear interpolation
pixels = scipy.interpolate.griddata(pts[:, :2], image.flatten(),
                                    pts_dist, method="linear")
pixels = np.round(pixels).astype(np.uint8)

image_undistorted = pixels.reshape((image.shape[0], image.shape[1]))

cv2.namedWindow("init")
cv2.imshow("init", image)

cv2.namedWindow("undistort")
cv2.imshow("undistort", image_undistorted)
