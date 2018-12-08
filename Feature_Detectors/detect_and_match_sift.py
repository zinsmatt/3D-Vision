#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter, maximum_filter



def gaussian_kernel(shape=(3,3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def detect_and_describe_SIFT(img, n_octaves=5, n_scales=3, base_sigma=1.6, C=0.5):
    """ returns a list of keypoints and their corresponding descriptors """
    per_octave_images = []
    dog_pyramids = []
    for o in range(n_octaves):
        dog = []
        prev = None
        cur = None
        scaled_img = cv2.resize(img, (0, 0), (0, 0), 1.0/(2**o), 1.0/(2**o), cv2.INTER_CUBIC)
        blurred_images_gradients = []
        for s in range(-1, n_scales+1):
            prev = cur
            sigma = 2**(s/n_scales) * base_sigma
            cur = gaussian_filter(scaled_img, sigma)
            # compute norm and orientation of the gradient
            if s >= 0 and s < n_scales:
                dy, dx = np.gradient(cur)
                norm = np.sqrt(dx**2+dy**2)
                angle = np.arctan2(dy, dx)
                blurred_images_gradients.append((norm, angle))
            if prev is not None:
                dog.append(np.abs(cur  - prev))
        dog = np.stack(dog, axis=2)
        dog_pyramids.append(dog)
        per_octave_images.append(blurred_images_gradients)

    # extract keypoints
    keypoints_per_oct = []
    for dog in dog_pyramids:
        max_dog = maximum_filter(dog, (3, 3, 3))
        is_kpt = np.logical_and(dog == max_dog , dog > C)
        is_kpt[:, :, 0] = 0
        is_kpt[:, :, -1] = 0
        y, x, s = np.where(is_kpt)
        kpts = np.vstack((x, y, s)).T
        keypoints_per_oct.append(kpts)

    # compute descriptors
    kps = []
    desc = []
    for o in range(n_octaves):
        for kp in keypoints_per_oct[o]:
            x, y, s = kp
            s -= 1
            norm_patch = per_octave_images[o][s][0][y-7:y+9, x-7:x+9]
            if norm_patch.shape[0] != 16 or norm_patch.shape[1] != 16:
                continue
            angle_patch = per_octave_images[o][s][1][y-7:y+9, x-7:x+9]
            norm_patch = norm_patch * gaussian_kernel((16, 16), 1.5*16)
            edges = np.linspace(-np.pi, np.pi, 9)
            descriptor = []
            for j in range(0, 16, 4):
                for i in range(0, 16, 4):
                    bins = np.zeros((8))
                    norm_subpatch = norm_patch[j:j+4, i:i+4]
                    angle_subpatch = angle_patch[j:j+4,i:i+4]
                    for b in range(len(edges)-1):
                        t = np.logical_and(angle_subpatch>=edges[b],
                                           angle_subpatch<edges[b+1])
                        bins[b] += np.sum(norm_subpatch[t])
                    t = angle_subpatch==edges[-1]
                    bins[-1] += np.sum(norm_subpatch[t])
                    descriptor.append(bins)
            descriptor = np.hstack(descriptor)
            descriptor /= np.linalg.norm(descriptor)
            kps.append([x*2**o, y*2**o])
            desc.append(descriptor)
    kps = np.vstack(kps)
    desc = np.vstack(desc)
    return kps, desc


def match_descriptors(desc1, desc2, max_ratio=0.7, match_threshold=0.6):
    """ matches two sets of descriptors. (desc2[i] <-> desc1[matches[i]]) """
    print("Matching...")
    matches = [-1] * desc2.shape[0]
    for i in range(desc2.shape[0]):
        diff = np.sum((desc2[i, :] - desc1)**2, axis=1)
        sorted_diff = np.argsort(diff)
        if diff[sorted_diff[0]] / diff[sorted_diff[1]] > max_ratio:
            continue
        if diff[sorted_diff[0]] > match_threshold:
            continue
        matches[i] = sorted_diff[0]
    return matches


if __name__ == "__main__":
    n_octaves = 5
    n_scales = 3
    base_sigma = 1.6
    C = 10
    max_ratio = 0.7 # ratio threshold for matching (o remove ambiguous match)
    match_threshold = 0.6

    img1 = cv2.imread("data/images/img_1.jpg", cv2.IMREAD_GRAYSCALE)
    img1 = img1.astype(np.float)
    img1 = cv2.resize(img1, (0, 0), (0, 0), 1.0/3, 1.0/3, cv2.INTER_CUBIC)


    img2 = cv2.imread("data/images/img_2.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = img2.astype(np.float)
    img2 = cv2.resize(img2, (0, 0), (0, 0), 1.0/3, 1.0/3, cv2.INTER_CUBIC)


    # detect keypoints and extract descriptors
    kps1, desc1 = detect_and_describe_SIFT(img1, n_octaves, n_scales, base_sigma, C)
    kps2, desc2 = detect_and_describe_SIFT(img2, n_octaves, n_scales, base_sigma, C)

    # match descriptors
    matches = match_descriptors(desc1, desc2, max_ratio, match_threshold)

    # convert image back to rgb to draw colored keypoints and matches
    img1 = img1.astype(np.uint8)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = img2.astype(np.uint8)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # draw matches
    keypoints1 = [cv2.KeyPoint(x, y, 1) for x, y in kps1]
    keypoints2 = [cv2.KeyPoint(x, y, 1) for x, y in kps2]
    out = None
    dmatches = []
    for i, v in enumerate(matches):
        if v >= 0:
            dmatches.append(cv2.DMatch(v, i, 1, 1))
    out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, dmatches, out,
                          (0, 255, 255), (0, 0, 255))

    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    cv2.imshow("win", out)
    cv2.imwrite("sift_matches.png", out)
