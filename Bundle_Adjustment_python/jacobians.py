import numpy as np


def deriv_wrt_t(K, R, t, X):
    uvs = K @ R @ (X + t)
    u = uvs[0]
    v = uvs[1]
    w = uvs[2]

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]
    du = np.array([fx*R[0, 0]+px*R[2, 0], fx*R[0, 1]+px*R[2, 1], fx*R[0, 2]+px*R[2, 2]])
    dv = np.array([fy*R[1, 0]+py*R[2, 0], fy*R[1, 1]+py*R[2, 1], fy*R[1, 2]+py*R[2, 2]])
    dw = R[2, :]

    deriv = np.array([[(w * du - u * dw) / w**2],
                      [(w * dv - v * dw) / w**2]])
    return deriv


def deriv_wrt_X(K, R, t, X):
    uvs = K @ R @ (X + t)
    u = uvs[0]
    v = uvs[1]
    w = uvs[2]

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]
    du = np.array([fx*R[0, 0]+px*R[2, 0], fx*R[0, 1]+px*R[2, 1], fx*R[0, 2]+px*R[2, 2]])
    dv = np.array([fy*R[1, 0]+py*R[2, 0], fy*R[1, 1]+py*R[2, 1], fy*R[1, 2]+py*R[2, 2]])
    dw = R[2, :]

    deriv = np.array([[(w * du - u * dw) / w**2],
                      [(w * dv - v * dw) / w**2]])
    return deriv

def deriv_wrt_R(K, R, t, X):
    uvs = K @ R @ (X + t)
    u = uvs[0]
    v = uvs[1]
    w = uvs[2]
    
    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]
    


    du = np.hstack([fx*(X+t), np.zeros(3), px*(X+t)])
    dv = np.hstack([fy*(X+t), np.zeros(3), py*(X+t)])
    dw = np.hstack([np.zeros(3), np.zeros(3), X+t])


    deriv = np.array([[(w * du - u * dw) / w**2],
                      [(w * dv - v * dw) / w**2]])
    return deriv

def deriv_wrt_q(K, q, t, X):
    qx, qy, qz, qw = q
    deriv = np.array([[0, -4*qy, -4*qz, 0],
                      [2*qy, 2*qx, -2*qw, -2*qz],
                      [2*qz, 2*qw, 2*qx, 2*qy],
                      [2*qy, 2*qx, 2*qw, 2*qz],
                      [-4*qx, 0, -4*qz, 0],
                      [-2*qw, 2*qz, 2*qy, 2*qx],
                      [2*qz, -2*qw, 2*qx, -2*qy],
                      [-4*qx, -4*qy, 0, 0],
                      [2*qw, 2*qz, 2*qy, 2*qx]])
    return deriv