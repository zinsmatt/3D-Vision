#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')




def get_rotation_around_axis(angle, axis):
    """ return the rotation matrix corresponding to a rotation "angle" around X, Y or Z """
    a = np.deg2rad(angle)
    if axis in "xX":
        return np.matrix([[1, 0, 0],
                         [0, np.cos(a), -np.sin(a)],
                         [0, np.sin(a), np.cos(a)]])
    elif axis in "yY":
        return np.matrix([[np.cos(a), 0, np.sin(a)],
                         [0,1,0],
                         [-np.sin(a), 0, np.cos(a)]])
    elif axis in "zZ":
        return np.matrix([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0],
                         [0, 0, 1]])
    else:
        print("Axis should be X, Y or Z")
    
def draw_camera(R, t, size, label):
    """ R and t are the camera orientation and camera position """
    print("Camera ", label)
    print(R)
    print(t)
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

    ax.scatter(pts_x[:,0], pts_x[:,1], pts_x[:,2], c="r")
    ax.scatter(pts_y[:,0], pts_y[:,1], pts_y[:,2], c="g")
    ax.scatter(pts_z[:,0], pts_z[:,1], pts_z[:,2], c="b")
    ax.text(t[0, 0], t[1, 0], t[2, 0], label)


P = np.loadtxt("data/p_W_corners.txt", delimiter=',')

minx, miny, minz = P.min(axis=0)
maxx, maxy, maxz = P.max(axis=0)

ax.set_xlim(min(0, minx), max(0, maxx))
ax.set_ylim(min(0, miny), max(0, maxy))
ax.set_zlim(min(0, minz), max(0, maxz))

ax.scatter(P[:, 0], P[:, 1],  P[:, 2], c="b")
    
R = np.eye(3)
t = np.zeros((3, 1))
draw_camera(R, t, 3, "origin")


Rt = np.array([[-0.01335806, -0.00278305, 0.01681191, 0.39499907],
               [ 0.00743037, -0.02010051, 0.00257485, 0.29955923],
               [-0.01499681, -0.00759781, -0.01322271, -0.86766805]])

R2 = Rt[:3, :3].T
t2 = (-R2.dot(Rt[:, 3])).reshape((3, 1)) * 100
  
#R2 = Rt[:3, :3]
#t2 = Rt[:, 3].reshape((3, 1)) * 100

draw_camera(R2, t2, 3, "cam1")
#plt.show()

