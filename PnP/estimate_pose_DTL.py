#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2

np.set_printoptions(suppress=True)

# estimate pose using DTL

# Load intrinsics, 3D points and 2D points
K = np.loadtxt("data/K.txt")
P = np.loadtxt("data/p_W_corners.txt", delimiter=',')
points_2d = np.loadtxt("data/detected_corners.txt")

# 2d points
p = points_2d[0, :].reshape((-1, 2))
p_hom = np.c_[p, np.ones((p.shape[0], 1))]
# normalized points (or calibrated points)
p_norm = np.linalg.inv(K).dot(p_hom.T).T

# 3D points
P_hom = np.c_[P, np.ones((P.shape[0], 1))]

# Build Q matrix
Q = np.zeros((P.shape[0] * 2, 12))
for i in range(P_hom.shape[0]):
    Q[i * 2, :4] = P_hom[i, :]
    Q[i * 2, 8:] = -p_norm[i, 0] * P_hom[i, :]
    Q[i * 2 + 1, 4:8] = P_hom[i, :]
    Q[i * 2 + 1, 8:] = -p_norm[i, 1] * P_hom[i, :]

# Solve Qx = 0
U, S, Vt = np.linalg.svd(Q)
solution = Vt[-1, :]
M = solution.reshape((3, 4))

# Extract R
R = M[:3, :3]
print("det = ", np.linalg.det(R))
print("Rt.R = ", R.T @ R)

# Reproject R on the SO(3) manifold
U_r, S_r, Vt_r = np.linalg.svd(R)
new_R = U_r @ Vt_r
if np.linalg.det(new_R) < 1:
    new_R *= -1
print("det = ", np.linalg.det(new_R))
print("Rt.R = ", new_R.T @ new_R)

# Estimate the scale factor
# It is implicitly given by the projection of the initial R
# on the manifold SO(3) and corresponds to the ratio of one of their norm.
scale_factor = np.linalg.norm(new_R) / np.linalg.norm(R)


# Final transformation
Rt = np.c_[new_R, scale_factor * M[:, 3]]

print("Rt \n", Rt)


# Load first image
image = cv2.imread("data/images_undistorted/img_0001.jpg", cv2.IMREAD_COLOR)

# Reproject 3D points
uvs = K @ Rt @ P_hom.T
uvs /= uvs[2, :]
uvs = uvs[:2, :].T

# Draw reprojections
for uv in np.round(uvs).astype(int):
    cv2.circle(image, tuple(uv), 2, (0, 0, 255), 1)

# Draw initial 2d points
for uv in np.round(p).astype(int):
    image[uv[1], uv[0], :] = (255, 0, 0)

# Compute the SSD error
print("SSD reprojection error = ",  np.sum(np.sum((uvs-p)**2, axis=1), axis=0))

# Display
cv2.namedWindow("fen")
cv2.imshow("fen", image[:, :, ::-1])