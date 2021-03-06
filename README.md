# 3D-Vision

Some 3D vision algorithms


### Augmented Reality
![alt augmented reality](Augmented_Reality/augmented_cube.gif "Augmented Cube")

## Camera Pose Estimation
<img src="Pose_Estimation/camera_poses.gif" width="375" height="237">

## Dense Stereo
![alt dense stereo](Reconstruction/disparity.gif "Dense Stereo Disparity")
<img src="Reconstruction/dense_pointcloud.png" width="540" height="200">


## Point Feature Detection
<img src="Feature_Detectors/detected_harris_corners.png" width="200" height="200"> <img src="Feature_Detectors/sift_matches.png" width="400" height="200">

![alt features tracking](Feature_Detectors/tracked_features.gif "Tracked Features")



## Pose Interpolation
The folder *Pose_Manipulation* contains code to interpolation between poses. This is done using spherical linear interpolation (SLERP) for the orientation and linear interpolation for the position.

<img src="Pose_Manipulation/pose_interpolation.gif" width="400" height="150">

<img src="Pose_Manipulation/pose_interpolation.png" width="400" height="150">


## Numercial Optimization
The folder *Numerical_Optimization* contains different algorithms for unconstrained non-linear minimization. For example, gradient descent or conjugate gradient.

<img src="Numerical_Optimization/doc/gradient_descent.png" width="310" height="310">
<img src="Numerical_Optimization/doc/nonlinear_least-squares_LM.png" width="400" height="300">


## Calibration
The folder *Calibration* contains algorithms to calibration the camera focal view from vanishing points.

<img src="Calibration/doc/calibration.png" width="426" height="320">
