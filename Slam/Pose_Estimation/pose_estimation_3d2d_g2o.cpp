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




class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
      _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
      Eigen::Matrix<double, 6, 1> u;
      u << update[0], update[1], update[2], 
           update[3], update[4], update[5];
      _estimate = Sophus::SE3d::exp(u) * _estimate;
    }

  virtual bool read(std::istream&) override {}
  virtual bool write(std::ostream&) const override {}
};

class EdgeProjection: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeProjection(const Eigen::Vector3d &X, const Eigen::Matrix3d& K): _X(X), _K(K) {}

  virtual void computeError() override
  {
    const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
    Sophus::SE3d Rt = v->estimate();
    Eigen::Vector3d p = _K * (Rt * _X);
    p /= p.z();
    _error = _measurement - p.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose*>(_vertices[0]);
    Sophus::SE3d Rt = v->estimate();
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    Eigen::Vector3d Xc = Rt * _X;
    double Z2 = std::pow(Xc.z(), 2);
    _jacobianOplusXi = Eigen::Matrix<double, 2, 6>::Zero();
    _jacobianOplusXi(0, 0) = -fx / Xc.z();
    _jacobianOplusXi(0, 2) = fx * Xc.x() / Z2;
    _jacobianOplusXi(0, 3) = fx * Xc.x() * Xc.y() / Z2;
    _jacobianOplusXi(0, 4) = -fx-fx * std::pow(Xc.x(), 2) / Z2;
    _jacobianOplusXi(0, 5) = fx * Xc.y() / Xc.z();

    _jacobianOplusXi(1, 1) = -fy / Xc.z();
    _jacobianOplusXi(1, 2) = fy * Xc.y() / Z2;
    _jacobianOplusXi(1, 3) = fy + fy * std::pow(Xc.y(), 2) / Z2;
    _jacobianOplusXi(1, 4) = -fy * Xc.y() * Xc.x() / Z2;
    _jacobianOplusXi(1, 5) = -fy * Xc.x() / Xc.z();
  }

  virtual bool read(std::istream&) override {}
  virtual bool write(std::ostream&) const override {}

  private:
    Eigen::Vector3d _X;
    Eigen::Matrix3d _K;
};


void pose_refinement_g2o(const VecVector3d& points_3d, const VecVector2d& points_2d, const Eigen::Matrix3d& K, Sophus::SE3d& pose)
{
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
  );
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  VertexPose* vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  for (int i = 0; i < points_3d.size(); ++i)
  {
    auto p2 = points_2d[i];
    auto p3 = points_3d[i];

    EdgeProjection* edge = new EdgeProjection(p3, K);
    edge->setId(i);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double >>(t2 - t1);
  cout << "optimization with g2o costs time: " << time_used.count() << " seconds." << endl;
  pose = vertex_pose->estimate();
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

  // cout << "calling bundle adjustment by gauss newton" << endl;
  // Sophus::SE3d pose_gn;
  // t1 = chrono::steady_clock::now();
  // bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  // t2 = chrono::steady_clock::now();
  // time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  // Eigen::Vector3d t_gn = pose_gn.translation();
  // Eigen::Matrix3d R_gn = pose_gn.so3().unit_quaternion().toRotationMatrix();
  // cout << "R_gn = " << endl << R_gn << endl;
  // cout << "t_gn = " << endl << t_gn << endl;

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
  cout << "Custom g2o" << endl;
  Sophus::SE3d pose_g2o;
  pose_refinement_g2o(pts_3d_eigen, pts_2d_eigen, K_eigen, pose_g2o);
  Eigen::Vector3d t_gn2 = pose_g2o.translation();
  Eigen::Matrix3d R_gn2 = pose_g2o.so3().unit_quaternion().toRotationMatrix();
  cout << "R_g2o = " << endl << R_gn2 << endl;
  cout << "t_g2o = " << endl << t_gn2 << endl;

  return 0;
}

