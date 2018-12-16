#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt


def compute_disparity(left, right, patch_radius=5, dmin=5, dmax=50):
    """ compute the disparity map from two stereo images left and right """
    # Skip one row over two for speed
    left = left.astype(np.float)
    right = right.astype(np.float)
    disp = np.zeros_like(left)
    for r in range(patch_radius, h-patch_radius, 2):        # skip 1 row over 2
        for c in range(patch_radius+dmax, w-patch_radius):
            patches = []
            for k in range(c-dmax, c-dmin+1):
                patches.append(right[r-patch_radius:r+patch_radius+1,
                                   k-patch_radius:k+patch_radius+1].flatten())
            patch = left[r-patch_radius:r+patch_radius+1,
                        c-patch_radius:c+patch_radius+1].flatten()
            patches = np.vstack(patches)
            diff = np.sum((patches-patch)**2, axis=1)
            sdiff = np.sort(diff)
            # filtering
            if sdiff[1] <= 1.5*sdiff[0] and sdiff[2] <= 1.5*sdiff[0]:
                continue
            d = dmax - np.argmin(diff)
            # filtering
            if d == dmax or d == dmin:
                continue
            disp[r, c] = d
        print(int(r * 100 / h), "%")
        
    # duplicate to fill the skipped rows
    for i in range(2, h, 2):
        disp[i, :] = disp[i-1, :]
    return disp


if __name__ == "__main__":
    out_dir = "disparity"
    names_left = sorted(glob.glob("data/left/*.png"))
    names_right = sorted(glob.glob("data/right/*.png"))
    for i in range(len(names_left)):
        filename = os.path.basename(names_left[i])
        im1 = cv2.imread(names_left[i], cv2.IMREAD_GRAYSCALE)  
        im2 = cv2.imread(names_right[i], cv2.IMREAD_GRAYSCALE)
        
        scale_factor = 0.5
        im1 = cv2.resize(im1, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        im2 = cv2.resize(im2, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        
        h, w = im1.shape
        dmax = 50
        dmin = 5
        patch_radius = 5
        
        disp = compute_disparity(im1, im2, patch_radius, dmin, dmax)
        
        plt.imshow(disp)
        disp /= dmax
        disp *= 255
        cv2.imwrite(os.path.join(out_dir, filename), disp)
