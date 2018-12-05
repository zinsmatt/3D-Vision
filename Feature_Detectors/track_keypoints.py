#!/usr/bin/env python3
"""
@author: matt
"""
import cv2
import numpy as np
import scipy.signal
import os
import glob


def extract_keypoints(img, ksize=9, k=0.04, n=200, dsize=25):
    """ Detect Harris corners and extract features
        (simple image patch around the keypoint) """
    # compute Harris response
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy

    box_filter = np.ones((ksize, ksize))

    s_Ix2 = scipy.signal.convolve2d(Ix2, box_filter, mode="same")
    s_Iy2 = scipy.signal.convolve2d(Iy2, box_filter, mode="same")
    s_IxIy = scipy.signal.convolve2d(Ix*Iy, box_filter, mode="same")

    R = s_Ix2 * s_Iy2 - s_IxIy * s_IxIy - k * (s_Ix2 + s_Iy2)

    # Extract keypoints and descriptors
    keypoints = []
    descriptors = []
    d_radius = dsize // 2
    R_big = cv2.copyMakeBorder(R, d_radius, d_radius, d_radius,
                               d_radius, cv2.BORDER_CONSTANT, 0)
    img_big = cv2.copyMakeBorder(img, d_radius, d_radius, d_radius,
                                 d_radius, cv2.BORDER_CONSTANT, 0)
    while len(keypoints) < n:
        max_idx = np.argmax(R_big.flatten())
        row = max_idx // R_big.shape[1]
        col = max_idx % R_big.shape[1]
        keypoints.append((col - d_radius, row - d_radius))    # -d_radius because of padding
        R_big[row-d_radius:row+d_radius+1, col-d_radius:col+d_radius+1] = 0
        patch = img_big[row-d_radius:row+d_radius+1, col-d_radius:col+d_radius+1]
        descriptors.append(patch.flatten())
#    plt.imshow(descriptors[0].reshape((dsize, dsize)))
    return np.vstack(keypoints), np.vstack(descriptors).astype(np.float)


def match_features(desc1, desc2, alpha=30):
    """ Match the two sets of features, assuming at least one match is possible.
        A feature in desc2 can match at most one feature of desc1 """
    matches = -np.ones((desc2.shape[0]), np.int)
    matches_ssd = np.zeros((desc2.shape[0]))
    for i in range(desc2.shape[0]):
        ssd = np.sum((desc1 - desc2[i, :])**2, axis=1)
        matches[i] = np.argmin(ssd)
        matches_ssd[i] = np.min(ssd)

    min_match_ssd = np.min(matches_ssd)
    matches[matches_ssd > alpha * min_match_ssd] = -1
    return matches


def draw_tracks(img, kp1, kp2, matches):
    """ matches is a list which contains, for each kp2, the index of
        the matched kp1 (-1 if no match) """
    for i, m in enumerate(matches):
        cv2.circle(img, tuple(kp2[i, :]), 4, (0, 255, 0), 2)
        if m >= 0:
            cv2.circle(img, tuple(kp1[m, :]), 4, (0, 255, 0), 2)
            cv2.line(img, tuple(kp2[i, :]), tuple(kp1[m, :]), (255, 0, 0), 2)


def track_first_frames():
    """ Simple example with only two frames """
    filename1 = "data/frames/000000.png"
    img1 = cv2.imread(filename1, cv2.IMREAD_UNCHANGED)
    kp1, desc1 = extract_keypoints(img1)

    filename2 = "data/frames/000002.png"
    img2 = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)
    kp2, desc2 = extract_keypoints(img2)

    matches = match_features(desc1, desc2)
    output_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    draw_tracks(output_img, kp1, kp2, matches)
    cv2.namedWindow("win")
    cv2.imshow("win", output_img)



if __name__ =="__main__":
    output_dir = "output"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    filenames = glob.glob("data/frames/*.png")
    filenames = sorted(filenames)

    for i in range(1, len(filenames)):
        filename1 = filenames[i-1]
        filename2 = filenames[i]
        output_filename = os.path.join(output_dir, os.path.basename(filename2))

        # read images
        img1 = cv2.imread(filename1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)

        # extract keypoints and descriptors
        kp1, desc1 = extract_keypoints(img1)
        kp2, desc2 = extract_keypoints(img2)

        # match descriptors
        matches = match_features(desc1, desc2)
        output_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # draw tracks
        draw_tracks(output_img, kp1, kp2, matches)

        # save result
        cv2.imwrite(output_filename, output_img)

        print("frames ", i-1, i)
    print("Done.")
