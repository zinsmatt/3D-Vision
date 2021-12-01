#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// void find_feature_matches(
//   const Mat &img_1, const Mat &img_2,
//   std::vector<KeyPoint> &keypoints_1,
//   std::vector<KeyPoint> &keypoints_2,
//   std::vector<DMatch> &matches);

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}


Point2d pixel2cam(const Point2d &p, const Mat &K);

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


// BA by gauss-newton
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

void pose_refinement_gauss_newton(const VecVector3d& points_3d, const VecVector2d& points_2d, const Eigen::Matrix3d& K, Sophus::SE3d& pose);

int main(int argc, char **argv) {
  // if (argc != 5) {
  //   cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
  //   return 1;
  // }
  string f1 = "../data/1.png"; //argv[1];
  string f2 = "../data/2.png"; //argv[2];
  string f3 = "../data/1_depth.png"; //argv[3];
  Mat img_1 = imread(f1, CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(f2, CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;


  Mat d1 = imread(f3, CV_LOAD_IMAGE_UNCHANGED);
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch m:matches) {
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
  Mat R;
  cv::Rodrigues(r, R);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;




  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }

  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  Eigen::Vector3d t_gn = pose_gn.translation();
  Eigen::Matrix3d R_gn = pose_gn.so3().unit_quaternion().toRotationMatrix();
  cout << "R_gn = " << endl << R_gn << endl;
  cout << "t_gn = " << endl << t_gn << endl;

  // cout << "calling bundle adjustment by g2o" << endl;
  // Sophus::SE3d pose_g2o;
  // t1 = chrono::steady_clock::now();
  // bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  // t2 = chrono::steady_clock::now();
  // time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;


  Eigen::Matrix3d K_eigen;
  K_eigen << 520.9, 0, 325.1,
             0, 521.0, 249.7,
             0, 0, 1;
  cout << "Custom Gauss-Newton" << endl;
  Sophus::SE3d pose_gn2(Sophus::SO3d(Eigen::Matrix3d::Identity()), Eigen::Vector3d::Zero());
  // Sophus::SE3d pose_gn2(Sophus::SO3d(Eigen::Matrix3d::Identity()), Eigen::Vector3d(0.5, -0.1, 0.2));
  t1 = chrono::steady_clock::now();
  pose_refinement_gauss_newton(pts_3d_eigen, pts_2d_eigen, K_eigen, pose_gn2);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp with custom Gauss-Newton cost time: " << time_used.count() << " seconds." << endl;
  
  Eigen::Vector3d t_gn2 = pose_gn2.translation();
  Eigen::Matrix3d R_gn2 = pose_gn2.so3().unit_quaternion().toRotationMatrix();
  cout << "R_gn2 = " << endl << R_gn2 << endl;
  cout << "t_gn2 = " << endl << t_gn2 << endl;

  return 0;
}

void pose_refinement_gauss_newton(const VecVector3d& points_3d, const VecVector2d& points_2d, const Eigen::Matrix3d& K, Sophus::SE3d& pose)
{
  const int max_iter = 100;
  double fx = K(0, 0);
  double fy = K(1, 1);
  const int n = points_2d.size();
  double prev_cost = 0.0;
  for (int it = 0; it < max_iter; ++it)
  {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    double cost = 0.0;
    for (int i = 0; i < n; ++i)
    {
      Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
      const auto& X = points_3d[i];
      Eigen::Vector3d x = K * (pose * X);
      x /= x.z();
      Eigen::Vector2d err = points_2d[i] - x.head(2);
      cost += err.squaredNorm();
      double Z2 = std::pow(X.z(), 2);
      J(0, 0) = -fx / X.z();
      J(0, 2) = fx * X.x() / Z2;
      J(0, 3) = fx * X.x() * X.y() / Z2;
      J(0, 4) = -fx-fx * std::pow(X.x(), 2) / Z2;
      J(0, 5) = fx * X.y() / X.z();

      J(1, 1) = -fy / X.z();
      J(1, 2) = fy * X.y() / Z2;
      J(1, 3) = fy + fy * std::pow(X.y(), 2) / Z2;
      J(1, 4) = -fy * X.y() * X.x() / Z2;
      J(1, 5) = -fy * X.x() / X.z();

      H += J.transpose() * J;
      b += -J.transpose() * err;
    }

    std::cout << "iter " << it << ": cost = " << cost << "\n";
    Eigen::Matrix<double, 6, 1> delta_x = H.inverse() * b;
    // Eigen::Matrix<double, 6, 1> delta_x = H.ldlt().solve(b);

    if (it > 0 && prev_cost - cost < 1e-5 || delta_x.norm() < 1e-5)
    {
      break;
    }

    // update pose
    pose = Sophus::SE3d::exp(delta_x) * pose;
    prev_cost = cost;
  }

}


Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

      Eigen::Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();
      Eigen::Matrix<double, 2, 6> J;
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
  }

  // cout << "pose by g-n: \n" << pose.matrix() << endl;
}
