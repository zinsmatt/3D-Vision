#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from estimate_pose_DLT import estimate_pose
import os
import glob

fig = plt.figure("Camera poses")
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')


def draw_camera(R, t, size, label):
    """ R and t are the camera orientation and camera position """
    # print("Camera ", label)
    # print(R)
    # print(t)
    new_x = R[:, 0].reshape((3, 1))
    new_y = R[:, 1].reshape((3, 1))
    new_z = R[:, 2].reshape((3, 1))

    pts_x = []
    for i in np.arange(0.0, size, 0.5):
        pts_x.append(t + i * new_x)
    pts_y = []
    for i in np.arange(0.0, size, 0.5):
        pts_y.append(t + i * new_y)
    pts_z = []
    for i in np.arange(0.0, size, 0.5):
        pts_z.append(t + i * new_z)
    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    pts_z = np.array(pts_z)

    ax.scatter(pts_x[:, 0], pts_x[:, 1], pts_x[:, 2], c="r")
    ax.scatter(pts_y[:, 0], pts_y[:, 1], pts_y[:, 2], c="g")
    ax.scatter(pts_z[:, 0], pts_z[:, 1], pts_z[:, 2], c="b")
    ax.text(t[0, 0], t[1, 0], t[2, 0], label)


# Load intrinsics, 3D points and 2D points
K = np.loadtxt("data/K.txt")
P = np.loadtxt("data/p_W_corners.txt", delimiter=',')
points_2d = np.loadtxt("data/detected_corners.txt")

ax.scatter(P[:, 0], P[:, 1],  P[:, 2], c="b")

# Draw origin
# draw_camera(np.eye(3), np.zeros((3, 1)), 5, "origin")


input_dir = "data/images_undistorted"

images_list = glob.glob(os.path.join(input_dir, "*.jpg"))
cam_positions = []
for i, f in enumerate(images_list):
    if i != 0 and i != 180 and i != 100:
        continue

    # Detected 2d points
    p = points_2d[i, :].reshape((-1, 2))

    # Estimate the pose of the first camera
    Rt = estimate_pose(K, p, P)

    orientation = Rt[:3, :3].T
    position = (-orientation.dot(Rt[:, 3])).reshape((3, 1))
    cam_positions.append(position)

    draw_camera(orientation, position, 10, "cam_"+str(i))

min_v = min(np.vstack(cam_positions).min(), P.min())
max_v = max(np.vstack(cam_positions).max(), P.max())

ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
ax.set_zlim(min_v, max_v)
