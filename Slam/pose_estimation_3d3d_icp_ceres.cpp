#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <chrono>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;

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

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}

// Solve ICP with linear algebra (SVD solution)
void pose_estimation_3d3d(vector<Eigen::Vector3d> pts1,
                          vector<Eigen::Vector3d> pts2,
                          Eigen::Matrix3d &R, Eigen::Vector3d &t) {
  Eigen::Vector3d c1(0.0, 0.0, 0.0), c2(0.0, 0.0, 0.0);
  for (auto& p : pts1)
    c1 += p;
  c1 /= pts1.size();
  for (auto& p : pts2)
    c2 += p;
  c2 /= pts2.size();

  for (auto& p : pts1)
    p -= c1;
  for (auto& p : pts2)
    p -= c2;
  
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < pts1.size(); ++i)
  {
    W += pts1[i] * pts2[i].transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0)
    R = -R;

  t = c1 - R * c2;
}



// Local parameterization needed to handle SE3 from Sophus (from Sophus/test/ceres/)
using namespace Sophus;
class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<SE3d const> const T(T_raw);
    Eigen::Map<Vector6d const> const delta(delta_raw);
    Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * SE3d::exp(delta);
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
        jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return SE3d::num_parameters; }

  virtual int LocalSize() const { return SE3d::DoF; }
};




struct ICPError
{
  ICPError(const Eigen::Vector3d& X1, const Eigen::Vector3d& X2)
    : _X1(X1), _X2(X2)
    {}

    template <class T>
    bool operator() (const T* const params, T* errors) const {
      const Eigen::Map<const Sophus::SE3<T>> Rt(params);
      Eigen::Map<Eigen::Matrix<T, 3, 1>> err(errors);
      err = _X1 - Rt * _X2;
      return true;
    }
  

  private:
    Eigen::Vector3d _X1;
    Eigen::Vector3d _X2;
};


// Solve ICP with non-linear optimization
void bundleAdjustment(
  const vector<Eigen::Vector3d> &pts1,
  const vector<Eigen::Vector3d> &pts2,
  Eigen::Matrix3d &R, Eigen::Vector3d &t) {

    Sophus::SE3d pose(R, t);
    ceres::Problem problem;
    for (int i = 0; i < pts1.size(); ++i)
    {
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ICPError, 3, 7>( // in g2o we use size 6 because its the size of the delta_x, here its the real size, the delta_x is handle in the local parameterization
          new ICPError(pts1[i], pts2[i])
        ),
        nullptr,
        pose.data()
      );
    }

    problem.SetParameterization(pose.data(), new LocalParameterizationSE3);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization ICP (ceres) costs time: " << time_used.count() << " seconds." << endl;
    std::cout << summary.BriefReport() << "\n";


    R = pose.so3().unit_quaternion().toRotationMatrix();
    t = pose.translation();
  }

int main(int argc, char **argv) {
  // if (argc != 5) {
  //   cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
  //   return 1;
  // }
  string f1 = "../data/1.png"; //argv[1];
  string f2 = "../data/2.png"; //argv[2];
  string f3 = "../data/1_depth.png"; //argv[3];
  string f4 = "../data/2_depth.png"; //argv[3];

  Mat img_1 = imread(f1, CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(f2, CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

  Mat depth1 = imread(f3, CV_LOAD_IMAGE_UNCHANGED);
  Mat depth2 = imread(f4, CV_LOAD_IMAGE_UNCHANGED);
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  Eigen::Matrix3d K_eigen;
  K_eigen << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
  std::vector<Eigen::Vector3d> pts1, pts2;

  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Eigen::Vector3d(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Eigen::Vector3d(p2.x * dd2, p2.y * dd2, dd2));
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t.transpose() << endl;
  // cout << "R_inv = " << R.t() << endl;
  // cout << "t_inv = " << -R.t() * t << endl;

  // verify p1 = R * p2 + t
  double total_error = 0.0;
  for (int i = 0; i < pts1.size(); i++) {
    total_error += (pts1[i] - (R * pts2[i] + t)).norm();
  }
  std::cout << "Mean error (SVD): " << total_error / pts1.size() << "\n";


  cout << "calling bundle adjustment" << endl;
  R = Eigen::Matrix3d::Identity();
  t = Eigen::Vector3d::Zero();
  bundleAdjustment(pts1, pts2, R, t);

  // verify p1 = R * p2 + t
  total_error = 0.0;
  for (int i = 0; i < pts1.size(); i++) {
    total_error += (pts1[i] - (R * pts2[i] + t)).norm();
  }
  std::cout << "Mean error (BA): " << total_error / pts1.size() << "\n";
}

