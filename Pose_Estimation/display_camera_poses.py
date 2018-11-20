#!/usr/bin/env python3
"""
@author: matt
"""

import numpy as np
import OpenGL.GL as gl
import pangolin
import glob
import os
import time
from estimate_pose_DLT import estimate_pose


class camera_display:
    def __init__(self, separate_thread=True):
        self.cameras = []
        self.cam_id = 0
        self.points = []

    def viewer_init(self):
        self.window = pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
            pangolin.ModelViewLookAt(-30, -10, -40, 25, 20, -10,
                                     pangolin.AxisDirection.AxisNegY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def viewer_thread(self, q=None):
        self.viewer_init()
        while not pangolin.ShouldQuit():
            self.viewer_refresh(q)
            time.sleep(0.05)

    def viewer_refresh(self, q):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # Draw Camera
        gl.glLineWidth(3)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCamera(self.cameras[self.cam_id], 2, 0.75, 2)

        # Draw previous positions
        prev_positions = []
        gl.glPointSize(1)
        gl.glLineWidth(1)
        gl.glColor3f(0.8, 0.8, 0.7)
        for i in range(0, self.cam_id+1):
            prev_positions.append(self.cameras[i][:3, 3])
        pangolin.DrawPoints(np.vstack(prev_positions))
        pangolin.DrawLine(np.vstack(prev_positions))

        self.cam_id = (self.cam_id + 1) % len(self.cameras)

        # Draw lines
        gl.glLineWidth(3)
        gl.glPointSize(6)
        colors = [(0.7, 0, 0), (0, 0.7, 0), (0.7, 0.7, 0)]
        for i in range(3):
            gl.glColor3f(*colors[i])
            pangolin.DrawPoints(P[i*4:i*4+4])
            for j, k in [(0, 1), (2, 3), (0, 2), (1, 3)]:
                pangolin.DrawLine(self.points[[i * 4 + j, i * 4 + k], :])

        pangolin.FinishFrame()


if __name__ == "__main__":
    K = np.loadtxt("data/K.txt")
    P = np.loadtxt("data/p_W_corners.txt", delimiter=',')
    points_2d = np.loadtxt("data/detected_corners.txt")

    input_dir = "data/images_undistorted"
    images_list = glob.glob(os.path.join(input_dir, "*.jpg"))
    cam_poses = []
    for i, f in enumerate(images_list):
        # Detected 2d points
        p = points_2d[i, :].reshape((-1, 2))
        # Estimate the pose of the first camera
        Rt = estimate_pose(K, p, P)
        # cam_poses.append(Rt)
        orientation = Rt[:3, :3].T
        position = (-orientation.dot(Rt[:, 3]))
        pose = np.eye(4)
        pose[:3, :3] = orientation
        pose[:3, 3] = position
        cam_poses.append(pose)

    display = camera_display()
    display.cameras = cam_poses
    print(cam_poses[0])
    print(P)
    display.points = P
    display.viewer_thread()
