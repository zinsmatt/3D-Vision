#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:48:57 2021

@author: mzins
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("data/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


img = cv2.imread("data/left12.jpg")
h,  w = img.shape[:2]
width = w
height = h
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult2.png',dst)


#%% Manual Undistort

K_inv = np.linalg.inv(mtx)

v, u = np.mgrid[:480, :640]
pts = np.vstack((u.flatten(), v.flatten(), np.ones((640*480))))
pts_n = K_inv @ pts
pts_n = pts_n.T

r2 = pts_n[:, 0]**2 + pts_n[:, 1]**2
r4 = r2**2
r6 = np.sqrt(r2)**6

k1, k2, p1, p2, k3 = dist.flatten().tolist()

f = 1 + k1 * r2 + k2*r4 + k3 * r6
x = pts_n[:, 0]
y = pts_n[:, 1]

x_dist = x * f + (2*p1*x*y + p2*(r2 + 2*x**2))
y_dist = y * f + (2*p1*(r2+2*y**2) + 2*p2*x*y)


pts_dist_n = pts_n.copy()
pts_dist_n[:, 0] = x_dist
pts_dist_n[:, 1] = y_dist
uvs_dist = (mtx @ pts_dist_n.T).T

# nearest-neighbour resampling
pixels = []
for uv in np.round(uvs_dist[:, :2]).astype(int):
    if uv[0] >= 0 and uv[0] < width and uv[1] >= 0 and uv[1] < height:
        pixels.append(img[uv[1], uv[0], 0])
    else:
        pixels.append(0)
out = np.vstack(pixels).reshape((-1, width))
cv2.imwrite('manual_calibresult_nearest.png', out)


# better resampling (linear interpolation)
map_dist = uvs_dist[:, :2].reshape((height, width, 2)).astype(np.float32)
out2 = cv2.remap(img, map_dist[:, :, 0], map_dist[:, :, 1], cv2.INTER_LINEAR)
cv2.imwrite('manual_calibresult2_linear_interp.png', out2)

