#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
import glob
import os

input_dir = "data/images"
output_dir = "data/undist_images/"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

k1, k2 = np.loadtxt("data/D.txt")
K = np.loadtxt("data/K.txt")
for f in glob.glob(os.path.join(input_dir, "*.jpg")):
    name = os.path.basename(f)
    image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    image_undistorted = cv2.undistort(image, K, np.array([k1, k2, 0, 0]))
    cv2.imwrite(os.path.join(output_dir, name), image_undistorted)
    print(name, " done")
