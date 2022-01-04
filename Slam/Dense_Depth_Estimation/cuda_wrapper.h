#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>



void update_cuda(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, cv::Mat depth, cv::Mat cov2);

bool epipolar_search(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr,
                     const Eigen::Vector2d& pt, double depth_mu, double depth_sigma2,
                     Eigen::Vector2d& best_pc, Eigen::Vector2d& epipolar_dir);

void update_depth_filter(const Eigen::Vector2d& pr, const Eigen::Vector2d& pc,
                         const Sophus::SE3d& Tcr, const Eigen::Vector2d& epipolar_dir, 
                         cv::Mat depth, cv::Mat cov2);
