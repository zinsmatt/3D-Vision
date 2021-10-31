import numpy as np
import cv2

k1 = -0.28340811
k2 = 0.07395907
p1 = 0.00019359
p2 = 1.76187114 * 10**-5
fx = 458.654
fy = 457.296
cx = 367.215
cy = 248.375

K = np.diag([fx, fy, 1.0])
K[0, 2] = cx
K[1, 2] = cy



img_file = "data/distorted.png"
img = cv2.imread(img_file)
h, w, _ = img.shape

undist_img = np.zeros((h, w), np.uint8)


# Slow version
# for i in range(h):
#     for j in range(w):
#         ii = (i - K[1, 2]) / K[1, 1]
#         jj = (j - K[0, 2]) / K[0, 0]
#         # print(i, j)

#         r = np.sqrt(ii**2 + jj**2)
#         jd = jj * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * jj * ii + p2 * (r * r + 2 * jj**2)
#         id = ii * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * ii**2) + 2 * p2 * jj*ii
#         jjj = int(round(jd * K[0, 0] + K[0, 2]))
#         iii = int(round(id * K[1, 1] + K[1, 2]))

#         if iii >= 0 and iii < h and jjj >= 0 and jjj < w:
#             undist_img[i, j] = img[iii, jjj, 0]


# Vectorized version
ii, jj = np.meshgrid(range(h), range(w))
pts = np.vstack((jj.flatten(), ii.flatten(), np.ones(h*w))).astype(int)

pts_n = np.linalg.inv(K) @ pts
r2 = np.sum(pts_n[:2, :]**2, axis=0)
r4 = r2**2

pts_n[:2, :] = pts_n[:2, :] * (1 + k1 * r2 + k2 * r4)
pts_n[0, :] += 2 * p1 * pts_n[0, :] * pts_n[1, :] + p2 * (r2 + 2 * pts_n[0, :]**2)
pts_n[1, :] += p1 * (r2 + 2 * pts_n[1, :]**2) + 2 * p2 * pts_n[0, :] * pts_n[1, :]
undist_pts = (K @ pts_n).T

undist_pts = np.round(undist_pts).astype(int)

goods = np.intersect1d(
    np.intersect1d(np.where(undist_pts[:, 0] >= 0)[0], np.where(undist_pts[:, 1] >= 0)[0]),
    np.intersect1d(np.where(undist_pts[:, 1] < h)[0], np.where(undist_pts[:, 0] < w)[0])
)

coords_dist = undist_pts[goods, :]
coords_undist = pts[:2, goods].T
undist_img[coords_undist[:, 1], coords_undist[:, 0]] = img[coords_dist[:, 1], coords_dist[:, 0], 0]

cv2.imwrite("undistorted.png", undist_img)