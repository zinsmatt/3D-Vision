#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import glob
import os

from estimate_pose_DLT import estimate_pose


# Load intrinsics, 3D points and 2D points
K = np.loadtxt("data/K.txt")
P = np.loadtxt("data/p_W_corners.txt", delimiter=',')
points_2d = np.loadtxt("data/detected_corners.txt")


input_dir = "data/images_undistorted"

images_list = glob.glob(os.path.join(input_dir, "*.jpg"))
for i, f in enumerate(images_list):

    # Detected 2d points
    p = points_2d[i, :].reshape((-1, 2))

    # Estimate the pose of the first camera
    Rt = estimate_pose(K, p, P)

    # Reproject 3D points
    P_hom = np.c_[P, np.ones((P.shape[0], 1))]
    uvs = K @ Rt @ P_hom.T
    uvs /= uvs[2, :]
    uvs = uvs[:2, :].T

    # Compute the SSD error
    print("SSD reprojection error = ",  np.sum(np.sum((uvs - p)**2, axis=1),
                                               axis=0))
