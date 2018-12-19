#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
import glob
import os


def compute_disparity(left, right, patch_radius=5, dmin=5, dmax=50):
    """ compute the disparity map from two stereo images left and right """
    # Skip one row over two for speed
    h, w = left.shape
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
            imin = np.argmin(diff)
            if imin == 0 or imin == len(diff)-1:
                continue
        
            poly = np.polyfit([imin-1, imin, imin+1], diff[imin-1:imin+2], 2)
            d = dmax  + poly[1] / (2*poly[0])
            disp[r, c] = d
        print(int(r * 100 / h), "%")

    # duplicate to fill the skipped rows
    for i in range(2, h, 2):
        disp[i, :] = disp[i-1, :]
    return disp


def triangulate_points(disp, K, baseline):
    """ triangulate points from disparity map """
    K_inv = np.linalg.inv(K)
    # get valid points coordinates in left and right
    y, x = np.mgrid[:disp.shape[0], :disp.shape[1]]
    x = x.flatten()
    y = y.flatten()
    d = disp.flatten()
    valid_disp = np.where(d)[0]
    pts0 = np.vstack((x, y, np.ones_like(x))).T.astype(np.float)
    pts0 = pts0[valid_disp, :]
    pts1 = pts0.copy()
    pts1[:, 0] -= d[valid_disp]

    # triangulate valid points
    pts0 = pts0.T
    pts1 = pts1.T
    A0 = K_inv.dot(pts0)
    A1 = -K_inv.dot(pts1)
    b = np.array([[baseline, 0, 0]]).T
    pts3d = []
    for i in range(A0.shape[1]):
        A = np.vstack((A0[:, i], A1[:, i])).T
        x = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
        pts3d.append(x[0] * A0[:, i])
    pts3d = np.vstack(pts3d)
    return pts3d


def get_color(pts3d, K, image):
    """ get the color of the 3d points from image """
    pts2d = K.dot(pts3d.T)
    pts2d /= pts2d[2, :]
    pts2d = pts2d.astype(int)
    colors = image[pts2d[1, :], pts2d[0, :]]
    return colors



if __name__ == "__main__":
    out_dir = "disparity"
    out_dir_pc = "pointclouds"
    names_left = sorted(glob.glob("data/left/*.png"))
    names_right = sorted(glob.glob("data/right/*.png"))
    
    scale_factor = 0.5
    baseline = 0.54
    dmax = 50
    dmin = 5
    patch_radius = 5
    
    K = np.loadtxt("data/K.txt")
    K[:2, :] *= scale_factor   # update intrinsics with the scale factor
    poses = np.loadtxt("data/poses.txt")
    
    for i in range(len(names_left)):
        filename = os.path.basename(names_left[i])
        im1 = cv2.imread(names_left[i], cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(names_right[i], cv2.IMREAD_GRAYSCALE)

        im1 = cv2.resize(im1, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        im2 = cv2.resize(im2, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)

        # compute disparity map
        disp = compute_disparity(im1, im2, patch_radius, dmin, dmax)

        # triangulate points from the disparity map
        pts3d = triangulate_points(disp, K, baseline)
        
        # get colors for 3D points using an image
        colors = get_color(pts3d, K, im1)
        
        # register the points using the pose
        Rt = poses[i, :].reshape((3, 4))
        pts3d = np.dot(Rt[:3, :3], pts3d.T) + Rt[:, 3].reshape((3, 1))
        pts3d = pts3d.T
        
        # write point cloud in obj file
        with open(os.path.join(out_dir_pc, filename)+".obj", "w") as fout:
            for i, p in enumerate(pts3d):
                c = str(colors[i])
                fout.write("v "+" ".join(map(str, p)) + " "
                           + c + " " + c + " " + c + "\n")
        # write disparity as grayscale image
        disp /= dmax
        disp *= 255
        cv2.imwrite(os.path.join(out_dir, filename), disp)
