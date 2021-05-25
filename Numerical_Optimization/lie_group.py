import numpy as np
from scipy.spatial.transform.rotation import Rotation as Rot


def to_lie_algebra(R):
    theta = np.arccos(0.5 * (np.trace(R) - 1))
    lnR = (theta / (2*np.sin(theta))) * (R - R.T)
    w = [lnR[2, 1], lnR[0, 2], lnR[1, 0]]
    return w


def skew_sym(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def to_lie_group(w):
    theta = np.linalg.norm(w)
    S = skew_sym(w)
    return np.eye(3) + np.sin(theta) / theta * S + (1-np.cos(theta)) / theta**2 * S @ S



R = Rot.from_euler("xyz", [5, -5, 10], degrees=True).as_matrix()
print("R init\n", R)

w = to_lie_algebra(R)

RR = to_lie_group(w)

print("R after\n", RR)





