#!/usr/bin/env python3
"""
@author: matt
"""

import glob

lines = []

for filename in glob.glob("pointclouds/*.obj"):
    with open(filename, "r") as fin:
        lines += fin.readlines()
        
with open("total_scene.obj", "w") as fout:
    for l in lines:
        fout.write(l)