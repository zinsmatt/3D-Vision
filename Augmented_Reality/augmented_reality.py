#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
import os
import glob

def axis_angle_to_matrix(v):
    """ Convert an axis-angle representation into a matrix_3x3
        representation of a rotation """
    k = v / np.linalg.norm(v)
    value = np.linalg.norm(v)
    kx = np.array([[0,    -k[2],  k[1]],
                   [k[2],     0, -k[0]],
                   [-k[1], k[0],    0]])
    R = np.eye(3) + np.sin(value) * kx + (1 - np.cos(value)) * kx.dot(kx)
    return R


# image = cv2.imread("data/images_undistorted/img_0001.jpg", cv2.IMREAD_COLOR)
#image = cv2.imread("data/undist_images/img_0001.jpg", cv2.IMREAD_COLOR)

input_dir = "data/undist_images"
output_dir = "data/AR"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# 3D model
square_size = 0.04
nb_x, nb_y = 9, 6
# generate points on grid
x, y = np.mgrid[:nb_x, :nb_y]
pts = np.c_[x.flatten(), y.flatten(), np.zeros((nb_x * nb_y, 1))]
pts *= square_size
lines = []

# generate points on cube
dim = np.array([2, 2, -2]) * square_size
origin = np.array([1, 0]) * square_size
cube = np.zeros((8, 3))
cube[1::2, 0] = dim[0]
cube[2:4, 1] = dim[1]
cube[6:, 1] = dim[1]
cube[4:, 2] = dim[2]
cube[:, 0] += origin[0]
cube[:, 1] += origin[1]

edges = [[0, 1], [2, 3], [0, 2], [1, 3],
         [4, 5], [6, 7], [4, 6], [5, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
pts = cube
lines = edges

# cube 2
shift = np.array([2, 0, 0]) * square_size
pts_2 = pts + shift
# cube 3
shift = np.array([0, 2, 0]) * square_size
pts_3 = pts + shift

pts_list = [pts, pts_2, pts_3]
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

# 3D CAD model
#points = []
#faces = []
#with open("data/teddy.obj", "r") as fin:
#    lines = fin.readlines()
#for l in lines:
#    if l[:2] == "v ":
#        points.append(list(map(float, l[2:].split(" "))))
#    elif l[:2] == "f ":
#        faces.append(list(map(int, l[2:].split(" "))))
#points = np.vstack(points)
#faces = np.vstack(faces) - 1
#edges = []
#for f in faces:
#    edges.append([f[0], f[1]])
#    edges.append([f[1], f[2]])
#    edges.append([f[2], f[0]])
#lines = edges
#pts = points

# Camera intrinsics
K = np.loadtxt("data/K.txt")

# Camera extrinsics
extrinsics = np.loadtxt("data/poses.txt")
list_of_images = glob.glob(os.path.join(input_dir, "*.jpg"))

for i, f in enumerate(sorted(list_of_images)):
    name = os.path.basename(f)
    image = cv2.imread(f, cv2.IMREAD_COLOR)

    # Get transformation from world to camera frame
    t = extrinsics[i, 3:]
    R = axis_angle_to_matrix(extrinsics[i, :3])
    # R = cv2.Rodrigues(extr[:3])[0] # opencv can be used
    Rt = np.hstack((R, t.reshape((-1, 1))))

    for i, pts in enumerate(pts_list):
        # 3D homogeneous points
        pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
        pts_h = pts_h.T
    
        # to camera frame
        pts_c = Rt.dot(pts_h)
    
        # projection at z=1
        pts_c_norm = pts_c / pts_c[2, :]
        # rescale by intrinsics
        uvs = K.dot(pts_c_norm)
    
        # Draw points and lines
        uvs = uvs[:2, :].astype(int).T
        for uv in uvs:
            cv2.circle(image, tuple(uv), 2, colors[i], 1)
    
        for edge in lines:
            cv2.line(image, tuple(uvs[edge[0]]), tuple(uvs[edge[1]]),
                     colors[i], 2, lineType=cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, name), image[:, :, ::-1])
    print(name, " done")

## with distortion correction (not needed here)
#xp = pts_c[0, :] / pts_c[2, :]
#yp = pts_c[1, :] / pts_c[2, :]
#
#k1, k2 = np.loadtxt("data/D.txt")
## k1, k2 = 0, 0
#
#r2 = xp**2 + yp**2
#
#factor = (1 + k1 * r2 + k2 * r2**2)
#xpp = xp * factor
#ypp = yp * factor
#
#norm_pts = np.vstack((xpp, ypp, np.ones_like(xpp)))
#uvs = K.dot(norm_pts)

# Display
#cv2.namedWindow("fen")
#cv2.imshow("fen", image)