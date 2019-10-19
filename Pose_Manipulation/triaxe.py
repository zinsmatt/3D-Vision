#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:00:28 2019

@author: Matthieu Zins
"""

import numpy as np


""" 
        Contains the functions to create and save a triaxe on disk
"""


def create_triaxe(size, sample, color=False):
    """ 
        Return the points of a triaxe. If color is True,
        colors are also returned
    """
    lin = np.linspace(0.0, size, sample)
    pts = np.zeros((3*sample, 3), dtype=np.float)
    pts[:sample, 0] = lin
    pts[sample:2*sample, 1] = lin
    pts[2*sample:3*sample, 2] = lin
    
    if color:
        colors = np.zeros((3*sample, 3), dtype=np.uint8)
        colors[:sample, 0] = 255
        colors[sample:2*sample, 1] = 255
        colors[2*sample:3*sample, 2] = 255
        return pts, colors
    else:
        return pts
        

def write_pointcloud_OBJ(filename, pts, colors=None):
    """ 
        Write a OBJ file from a pointcloud (Nx3) and optional colors (Nx3)
    """
    assert pts.shape[1] == 3
    with open(filename, "w") as fout:
        for i, p in enumerate(pts):
            s = "v " + " ".join(map(str, p))
            if colors is not None:
                s += " " + " ".join(map(str, colors[i]))
            s += "\n"
            fout.write(s)
        


     
def write_pointcloud_PLY(filename, pts, colors=None):
    """
        Write a PLY file from a pointcloud (Nx3) and optional colors (Nx3)
    """
    assert pts.shape[1] == 3
    ply_header = """ply
format ascii 1.0
comment VCGLIB generated
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 0
property list uchar int vertex_indices
end_header
""" % pts.shape[0]
    with open(filename, "w") as fout:
        fout.write(ply_header)
        for i, p in enumerate(pts):
            s = " ".join(map(str, p))
            if colors is not None:
                s += " " + " ".join(map(str, colors[i]))
            else:
                s += " 0 0 0\n"
            s += "\n"
            fout.write(s)
