#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// void find_feature_matches(
//   const Mat &img_1, const Mat &img_2,
//   std::vector<KeyPoint> &keypoints_1,
//   std::vector<KeyPoint> &keypoints_2,
//   std::vector<DMatch> &matches);



Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

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


typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


struct ProjectionError
{
  ProjectionError(const Eigen::Vector2d& measurement, const Eigen::Vector3d& point,
                  const Eigen::Matrix3d& K) : _x(measurement), _X(point), _K(K)
    {}

  template <class T>
  bool operator() (const T* const params, T* residuals) const
  {
    T X[3] = {T(_X.x()), T(_X.y()), T(_X.z())};
    T rot_X[3];
    ceres::UnitQuaternionRotatePoint(params, X, rot_X);
    rot_X[0] += params[4];
    rot_X[1] += params[5];
    rot_X[2] += params[6];
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    
    // T fx = T(_K(0, 0));
    // T fy = T(_K(1, 1));
    // T cx = T(_K(0, 2));
    // T cy = T(_K(1, 2));

    residuals[0] = _x[0] - ((rot_X[0] / rot_X[2]) * fx + cx); // parentheses have importance but do not known why
    residuals[1] = _x[1] - ((rot_X[1] / rot_X[2]) * fy + cy);


    // T rot[9];
    // ceres::QuaternionToRotation(params, rot);
    // Eigen::Matrix<T, 3, 3> R;
    // R << rot[0], rot[1], rot[2],
    //      rot[3], rot[4], rot[5],
    //      rot[6], rot[7], rot[8];

    // Eigen::Matrix<T, 3, 1> p(T(_X.x()), T(_X.y()), T(_X.z()));
    // Eigen::Matrix<T, 3, 1> t(params[4], params[5], params[6]);

    // Eigen::Matrix<T, 3, 3> K;
    // K << T(_K(0, 0)),  T(_K(0, 1)),  T(_K(0, 2)),
    //      T(_K(1, 0)),  T(_K(1, 1)),  T(_K(1, 2)),
    //      T(_K(2, 0)),  T(_K(2, 1)),  T(_K(2, 2));
    // Eigen::Matrix<T, 3, 1> p2 = K * (R * p + t);

    // residuals[0] = _x[0] - p2.x()/p2.z();
    // residuals[1] = _x[1] - p2.y()/p2.z();

    return true;
  }

  private:
    Eigen::Vector2d _x;
    Eigen::Vector3d _X;
    Eigen::Matrix3d _K;
};




void pose_refinement_ceres(const VecVector3d& points_3d, const VecVector2d& points_2d, const Eigen::Matrix3d& K, Sophus::SE3d& pose)
{
  Eigen::Vector3d trans = pose.translation();
  Eigen::Quaterniond quat = pose.so3().unit_quaternion();

  double pose_params[7] = {
    quat.w(), quat.x(), quat.y(), quat.z(), trans.x(), trans.y(), trans.z()
  };

  ceres::Problem problem;
  for (int i = 0; i < points_3d.size(); ++i)
  {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<ProjectionError, 2, 7>(
      // new ceres::NumericDiffCostFunction<ProjectionError, ceres::CENTRAL, 2, 7>(
        new ProjectionError(points_2d[i], points_3d[i], K)
      ),
      nullptr,
      pose_params
    );
  }
  auto *quat_t_parameterization = new ceres::ProductParameterization(new ceres::QuaternionParameterization(),
                                                                     new ceres::IdentityParameterization(3));
  problem.SetParameterization(pose_params, quat_t_parameterization);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double >>(t2 - t1);
  cout << "optimization with ceres costs time: " << time_used.count() << " seconds." << endl;
  std::cout << summary.BriefReport() << std::endl;

  Eigen::Quaterniond q(pose_params[0], pose_params[1], pose_params[2], pose_params[3]);
  Eigen::Vector3d t(pose_params[4], pose_params[5], pose_params[6]);

  pose = Sophus::SE3d(q.toRotationMatrix(), t);
}



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
  Eigen::Matrix3d K_eigen;
  K_eigen << 520.9, 0, 325.1,
             0, 521.0, 249.7,
             0, 0, 1;

  // Ceres
  cout << "Custom Ceres" << endl;
  Sophus::SE3d pose_ceres;
  pose_refinement_ceres(pts_3d_eigen, pts_2d_eigen, K_eigen, pose_ceres);
  Eigen::Vector3d t_ceres = pose_ceres.translation();
  Eigen::Matrix3d R_ceres = pose_ceres.so3().unit_quaternion().toRotationMatrix();
  cout << "R_ceres = " << endl << R_ceres << endl;
  cout << "t_ceres = " << endl << t_ceres << endl;

  return 0;
}

