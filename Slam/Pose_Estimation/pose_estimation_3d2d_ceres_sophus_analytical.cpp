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


using namespace Sophus;
// Local parameterization needed to handle SE3 from Sophus (from Sophus/test/ceres/)
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



// struct ProjectionError
// {
//   ProjectionError(const Eigen::Vector2d& measurement, const Eigen::Vector3d& point,
//                   const Eigen::Matrix3d& K) : _x(measurement), _X(point), _K(K)
//     {}

//   template <class T>
//   bool operator() (const T* const params, T* residuals) const
//   {
//     const Eigen::Map<const Sophus::SE3<T>> Rt(params);
//     Eigen::Matrix<T, 3, 1> X(T(_X.x()), T(_X.y()), T(_X.z()));
//     Eigen::Matrix<T, 3, 1> uv = _K * (Rt * X);
//     residuals[0] = _x[0] - uv.x() / uv.z();
//     residuals[1] = _x[1] - uv.y() / uv.z();
//     // std::cout << "residuals: \n";
//     // std::cout << residuals[0] << std::endl;
//     // std::cout << residuals[1] << std::endl << std::endl;;

//     // std::cout << T(residuals[0]) << "\n";
//     // std::cout << T(residuals[1]) << "\n";
//     return true;
//   }

//   private:
//     Eigen::Vector2d _x;
//     Eigen::Vector3d _X;
//     Eigen::Matrix3d _K;
// };


class ProjectionError: public ceres::SizedCostFunction<2, 7>
{
  public:
    ProjectionError(const Eigen::Vector2d& measurement, const Eigen::Vector3d& point,
                  const Eigen::Matrix3d& K) : _x(measurement), _X(point), _K(K) {}

  virtual bool Evaluate(double const * const * params,
                        double *residuals,
                        double **jacobians) const
  {
    const Eigen::Map<const Sophus::SE3d> Rt(params[0]);
    Eigen::Vector3d Xc = Rt * _X;
    Eigen::Vector3d Xp = Xc / Xc.z();
    Eigen::Vector3d uv = _K * Xp;
    residuals[0] = _x[0] - uv.x();
    residuals[1] = _x[1] - uv.y();

    if (!jacobians)
      return true;

    double *J = jacobians[0];
    if (!J)
      return true;

 
    double x = Xc.x();
    double y = Xc.y();
    double z = Xc.z();
    double z2 = z * z;

    Eigen::Matrix2d dedxp; // derivate wrt. x projected
    dedxp << -_K(0, 0), 0.0,
             0.0, -_K(1, 1);

    Eigen::Matrix<double, 2, 3> dxpdXc; // derivative of x projected wrt. Xcam
    dxpdXc << 1.0/z, 0.0, -x/z2,
              0.0, 1.0/z, -y/z2;

    double qx = params[0][0];
    double qy = params[0][1];
    double qz = params[0][2];
    double qw = params[0][3];
    double x3 = _X.x();
    double y3 = _X.y();
    double z3 = _X.z();

    Eigen::Matrix<double, 3, 4> dXcdq; // derivative of Xcam wrt. quaternions
    dXcdq(0, 0) = 2*qy*y3 + 2*qz*z3;
    dXcdq(0, 1) = 2*qw*z3 + 2*qx*y3 - 4*qy*x3;
    dXcdq(0, 2) = -2*qw*y3 + 2*qx*z3 - 4*qz*x3;
    dXcdq(0, 3) = 2*qy*z3 - 2*qz*y3;

    dXcdq(1, 0) = -2*qw*z3 - 4*qx*y3 + 2*qy*x3;
    dXcdq(1, 1) = 2*qx*x3 + 2*qz*z3;
    dXcdq(1, 2) = 2*qw*x3 + 2*qy*z3 - 4*qz*y3;
    dXcdq(1, 3) = -2*qx*z3 + 2*qz*x3;

    dXcdq(2, 0) = 2*qw*y3 - 4*qx*z3 + 2*qz*x3;
    dXcdq(2, 1) = -2*qw*x3 - 4*qy*z3 + 2*qz*y3;
    dXcdq(2, 2) = 2*qx*x3 + 2*qy*y3;
    dXcdq(2, 3) = 2*qx*y3 - 2*qy*x3;


    Eigen::Matrix3d dXcdt = Eigen::Matrix3d::Identity();  // derivative of Xcam wrt. translation

    
    Eigen::Matrix<double, 2, 7> J_tmp;
    J_tmp.block<2, 4>(0, 0) = dedxp * dxpdXc * dXcdq;
    J_tmp.block<2, 3>(0, 4) = dedxp * dxpdXc * dXcdt;

    for (int i = 0; i < 7; ++i)
    {
      J[i] = J_tmp(0, i);
      J[i+7] = J_tmp(1, i);
    }
    // std::cout << m << "\n";

    return true;    
  }


  private:
    Eigen::Vector2d _x;
    Eigen::Vector3d _X;
    Eigen::Matrix3d _K;
};





void pose_refinement_ceres(const VecVector3d& points_3d, const VecVector2d& points_2d, const Eigen::Matrix3d& K, Sophus::SE3d& pose)
{
  ceres::Problem problem;
  for (int i = 0; i < points_3d.size(); ++i)
  {
    problem.AddResidualBlock(
      new ProjectionError(points_2d[i], points_3d[i], K),
      nullptr,
      pose.data()
    );
  }
  problem.AddParameterBlock(pose.data(), 7, new LocalParameterizationSE3());

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
}



int main(int argc, char **argv) {
  // if (argc != 5) {
  //   cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
  //   return 1;
  // }
  string f1 = "../1.png"; //argv[1];
  string f2 = "../2.png"; //argv[2];
  string f3 = "../1_depth.png"; //argv[3];
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

