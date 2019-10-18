#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 23:22:06 2019

@author: Matthieu Zins
"""

"""
        This example shows how to interpolate between two poses using 
        SLERP for rotation and linear interpolation for the position.
"""

import numpy as np
from triaxe import create_triaxe, write_pointcloud_PLY
from pose_interpolation import apply_pose, interpolate_pose

triaxe, color = create_triaxe(1, 30, True)

pose1 = np.array([-0.2449494, 0.3877961, 0.0378426, 0.8877961, 0.0, 0.0, 0.0])
pose2 = np.array([0.8504829, -0.2251278, 0.4502556, -0.1525327, 3.0, -1.0, 5.0])

triaxe1 = apply_pose(pose1, triaxe)
triaxe2 = apply_pose(pose2, triaxe)

write_pointcloud_PLY("pose_start.ply", triaxe1, color)
write_pointcloud_PLY("pose_end.ply", triaxe2, color)

for i, t in enumerate(np.linspace(0, 1, 30)):
    pose_t = interpolate_pose(t, 0.0, pose1, 1.0, pose2)
    triaxe_t = apply_pose(pose_t, triaxe)
    write_pointcloud_PLY("pose_%d.ply" % i, triaxe_t, color)
