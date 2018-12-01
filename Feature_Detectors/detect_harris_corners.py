#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
import scipy.signal


def harris_corners(image, window_size=5, k=0.09, threshold=100000,
                   non_maxima_box_size=-1, nb_corners=-1):

    # to grayscale and float
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float)

    Iy, Ix = np.gradient(img)
    Ix2 = np.power(Ix, 2)
    Iy2 = np.power(Iy, 2)
    box_filter = np.ones((window_size, window_size))

    s_Ix2 = scipy.signal.convolve2d(Ix2, box_filter, mode="same")
    s_Iy2 = scipy.signal.convolve2d(Iy2, box_filter, mode="same")
    s_IxIy = scipy.signal.convolve2d(Ix*Iy, box_filter, mode="same")

    # Compute Harris response
    R = s_Ix2 * s_Iy2 - s_IxIy * s_IxIy - k * (s_Ix2+s_Iy2)**2

    # Threshold
    R_thres = R
    R_thres[R_thres < threshold] = 0

    # Non-maxima suppression
    if non_maxima_box_size > 0:
        for j in np.arange(0, R_thres.shape[0], non_maxima_box_size):
            for i in np.arange(0, R_thres.shape[1], non_maxima_box_size):
                box = R_thres[j:min(R_thres.shape[0], j + non_maxima_box_size),
                              i:min(R_thres.shape[1], i + non_maxima_box_size)]
                max_R = np.max(box)
                if (max_R > 0):
                    max_pos = np.where(box == max_R)
                    box[:, :] = 0
                    box[max_pos[0][0], max_pos[1][0]] = max_R

    # Extract corners and sort them by response magnitude
    corners = np.where(R_thres > 0)
    values = R_thres[corners]
    corners = np.vstack(corners[::-1]).T

    sorted_corners = np.argsort(values)[::-1]
    if (nb_corners > 0):
        return corners[sorted_corners[:nb_corners], :]
    else:
        return corners[sorted_corners, :]


if __name__ == "__main__":
    image = "data/checkerboard.png"

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    corners = harris_corners(img, nb_corners=16, non_maxima_box_size=25)

    for c in corners:
        cv2.circle(img, tuple(c), 8, (0, 0, 255), 2)

    cv2.namedWindow("win")
    cv2.imshow("win", img)
    cv2.imwrite("detected_harris_corners.png", img)
