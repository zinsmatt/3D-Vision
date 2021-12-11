import numpy as np
from numpy.testing._private.utils import measure
from scipy import sparse
import cv2
from ellcv.io import write_ply
from ellcv.utils import look_at, pose_error
from ellcv.visu import draw_points, generate_triaxe_pointcloud

from scipy.spatial.transform.rotation import Rotation as Rot
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from jacobians import *
# np.random.seed(1994)

# Generate random points
N = 1000
scene_min = -1.0
scene_max = 1.0
points = np.random.uniform(scene_min, scene_max, size=(N, 3))
points = np.loadtxt("chair2.xyz")
points = points[::5, :]
# points = points[:10, :]
N = points.shape[0]
points_h = np.hstack((points, np.ones((points.shape[0], 1))))


# Generate random camera
W, H = 640, 480
f = 550.0
K = np.array([[f, 0.0, W/2],
              [0.0, f, H/2],
              [0.0, 0.0, 1.0]])
T = 4
cam_dist_min = 4.0
cam_dist_max = 8.0
cam_z_min = -2.0
cam_z_max = 2.5
cam_target_min = -0.5
cam_target_max = 0.5
cameras_gt = []
for c in range(T):
    dist = np.random.uniform(cam_dist_min, cam_dist_max, 1)
    theta = np.random.uniform(-np.pi, np.pi, 1)
    x = np.cos(theta) * dist
    y = np.sin(theta) * dist
    z = np.random.uniform(cam_z_min, cam_z_max, 1)
    pos = np.asarray([x, y, z]).flatten()
    target = np.random.uniform(cam_target_min, cam_target_max, size=(3))
    o = look_at(pos, target)
    R = o.T
    t = -o.T.dot(pos)
    cameras_gt.append([R, t, K @ np.hstack((R, t.reshape((-1, 1))))])




# Add noise to camera poses
cameras_noisy = []
for ci, (R, t, P) in enumerate(cameras_gt):
    # t_noisy = t + np.random.uniform(-0.2, 0.2, size=(3))
    # R_noisy = Rot.from_euler("xyz", np.random.uniform(-1.0, 1.0, size=(3)), degrees=True).as_matrix() @ R
    t_noisy = t + np.random.randn(3) * 0.2
    R_noisy = Rot.from_euler("xyz", np.random.randn(3), degrees=True).as_matrix() @ R
    cameras_noisy.append([R_noisy, t_noisy, K @ np.hstack((R_noisy, t_noisy.reshape((-1, 1))))])


# Initial Reconstruction
def is_inside(p):
    return np.logical_and(np.logical_and(p[:, 0] >= 0, p[:, 0] < W),
                          np.logical_and(p[:, 1] >= 0, p[:, 1] < H))


# Project
gt_projections = []
gt_valid = []
projections_noisy_pose = []
measurements_noisy = []
for ci, (R, t, P) in enumerate(cameras_gt):
    uvs = P @ points_h.T
    uvs /= uvs[2, :]
    uvs =  uvs[:2, :].T
    gt_valid.append(is_inside(uvs))
    gt_projections.append(uvs)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    draw_points(img, uvs, color=(0, 0, 255))

    P_noisy = cameras_noisy[ci][2]
    uvs_noisy_pose = P_noisy @ points_h.T
    uvs_noisy_pose /= uvs_noisy_pose[2, :]
    uvs_noisy_pose =  uvs_noisy_pose[:2, :].T
    projections_noisy_pose.append(uvs_noisy_pose)
    # draw_points(img, uvs_noisy_pose, color=(255, 0, 0))

    # noisy points observation in image
    noisy_uvs = uvs# + np.random.randn(uvs.shape[0], 2) * 3*0
    measurements_noisy.append(noisy_uvs)
    draw_points(img, noisy_uvs, color=(0, 255, 0))

    cv2.imshow("viz", img)
    cv2.waitKey()




def display_reproj(measurements, projections, points, name):
    for i, (meas, P) in enumerate(zip(measurements, projections)):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        draw_points(img, meas, color=(0, 0, 255))
        uvs = P @ points_h.T
        uvs /= uvs[2, :]
        uvs = uvs[:2, :].T
        draw_points(img, uvs)
        error = np.mean(np.sqrt(np.sum((meas - uvs)**2, axis=1)))
        # cv2.imshow("reproj", img)
        # cv2.waitKey()
        cv2.putText(img, "%.4f" % error, (30, 30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(name + "_%03d.png" % i, img)


display_reproj(measurements_noisy, [P for R, t, P in cameras_noisy], points, "before/img")


# Save triaxes
gt_triaxes_pts = []
gt_triaxes_col = []
for R, t, P in cameras_gt:
    pts, col = generate_triaxe_pointcloud([R.T, -R.T@t], size=0.5, sampling=100)
    gt_triaxes_pts.append(pts)
    gt_triaxes_col.append(col)

write_ply("triaxes_gt.ply", np.vstack(gt_triaxes_pts), np.vstack(gt_triaxes_col))

noisy_triaxes_pts = []
noisy_triaxes_col = []
for R, t, P in cameras_noisy:
    pts, col = generate_triaxe_pointcloud([R.T, -R.T@t], size=0.5, sampling=100)
    noisy_triaxes_pts.append(pts)
    noisy_triaxes_col.append(col)

write_ply("triaxes_noisy.ply", np.vstack(noisy_triaxes_pts), np.vstack(noisy_triaxes_col))



# Triangulate points
print(N, "points")
triangulated = np.ones((N, 3), dtype=float) * -6
for i in range(N):
    cam_indices = []
    keypoints = []
    projections = []
    for j in range(T):
        if gt_valid[j][i]:
            cam_indices.append(j)
            keypoints.append(gt_projections[j][i, :2].reshape((2, 1)))
            projections.append(cameras_noisy[j][2])
        if len(cam_indices) >= 2:
            break
    
    if len(cam_indices) < 2:
        print("Warning: could not triangulate the point (not visible in at least 2 images")
        continue
    X = cv2.sfm.triangulatePoints(keypoints, projections)
    triangulated[i, :] = X.flatten()

write_ply("triangulated.ply", triangulated)
triangulated_h = np.hstack((triangulated, np.ones((triangulated.shape[0], 1))))





##### Pose optimization #####
# We minimize the reprojection error between the model points and the retected ones.


# def to_lie_algebra(R):
#     theta = np.arccos(0.5 * (np.trace(R) - 1))
#     lnR = (theta / (2*np.sin(theta))) * (R - R.T)
#     w = [lnR[2, 1], lnR[0, 2], lnR[1, 0]]
#     return w


# def skew_sym(w):
#     return np.array([[0, -w[2], w[1]],
#                      [w[2], 0, -w[0]],
#                      [-w[1], w[0], 0]])

# def to_lie_group(w):
#     theta = np.linalg.norm(w)
#     S = skew_sym(w)
#     return np.eye(3) + np.sin(theta) / theta * S + (1-np.cos(theta)) / theta**2 * S @ S


# Build state vector
params = []
for R, t, P in cameras_noisy:
    # r = to_lie_algebra(R)
    r = Rot.from_matrix(R).as_quat().tolist()
    params.extend(r + t.tolist())
params += triangulated.flatten().tolist()

print(len(params), "parameters")

print("valid = ", np.mean(gt_valid))

def callback_least_squares(x, measurements, measurements_valid, points_h, n_cams):
    Ps = []
    for i in range(0, n_cams*7, 7):
        R = Rot.from_quat(x[i:i+4]).as_matrix()
        t = np.asarray(x[i+4:i+7]).reshape((3, 1))
        Ps.append(K @ np.hstack((R, t)))

    points_h = x[7*n_cams:].reshape((-1, 3))
    points_h = np.hstack((points_h, np.ones((points_h.shape[0], 1))))

    errors = []
    for ci, P in enumerate(Ps):
        uvs = P @ points_h.T
        uvs /= uvs[2, :]
        uvs = uvs[:2, :].T
        d = uvs - measurements[ci]
        # d[:, 0] *= measurements_valid[ci]
        # d[:, 1] *= measurements_valid[ci]
        errors.append(d.flatten().tolist())
    errors =  np.hstack(errors)
    return errors


def ba_sparsity(n_params, n_cameras, n_points, measurements_valid):

    A = lil_matrix((n_points * n_cameras * 2, n_params), dtype=int)
    
    k = n_points*2
    for ci in range(n_cameras):
        # A[ci*k:ci*k+k, ci*7:ci*7+7] = np.repeat(measurements_valid[ci].reshape((-1, 1)), 7, axis=1)

        for pi in range(n_points):
            # cameras
            A[ci*k+pi*2, ci*7:ci*7+7] = measurements_valid[ci][pi]
            A[ci*k+pi*2+1, ci*7:ci*7+7] = measurements_valid[ci][pi]

            # points
            A[ci*k+pi*2, n_cameras*7+pi*3:n_cameras*7+pi*3+3] = measurements_valid[ci][pi]
            A[ci*k+pi*2+1, n_cameras*7+pi*3:n_cameras*7+pi*3+3] = measurements_valid[ci][pi]

    return A

sparse_jacobian = ba_sparsity(len(params), T, N, gt_valid)


# for l in sparse_jacobian.todense():
#     print(l.flatten().tolist())

# res = least_squares(callback_least_squares, params, args=(measurements_noisy, gt_valid, points_h, T))
res = least_squares(callback_least_squares, params, args=(measurements_noisy, gt_valid, triangulated_h, T), jac_sparsity=sparse_jacobian)
x_optim = res.x

cameras_optim = []
optim_cam_pts = []
optim_cam_col = []
for i in range(0, T*7, 7):
    R = Rot.from_quat(x_optim[i:i+4]).as_matrix()
    t = np.asarray(x_optim[i+4:i+7]).reshape((3, 1))
    cameras_optim.append((R, t, K @ np.hstack((R, t))))
    pts, col = generate_triaxe_pointcloud([R.T, -R.T @ t], size=0.7, sampling=100)
    optim_cam_pts.append(pts)
    optim_cam_col.append(col)

write_ply("optim_cameras.ply", np.vstack(optim_cam_pts), np.vstack(optim_cam_col))

def invert_pose(pose):
    return pose[0].T, -pose[0].T.dot(pose[1])


print("=================== Results ================")
for ci in range(T):
    rot_err, pos_err = pose_error(invert_pose(cameras_optim[ci]), invert_pose(cameras_gt[ci]))
    print("Camera", ci)
    print("\trot error = %.4f°" % np.rad2deg(rot_err))
    print("\tpos error = %.4fm" % pos_err)
    print()


optim_points = x_optim[7*T:].reshape((-1, 3))
write_ply("optim_points.ply", np.vstack(optim_points))



print(N)
print(sparse_jacobian.shape)

display_reproj(measurements_noisy, [P for R, t, P in cameras_optim], optim_points, "after/img")