#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
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

class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      
    virtual void setToOriginImpl() override {
      _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override {
      Eigen::Matrix<double, 6, 1> u;
      u << update[0], update[1], update[2], update[3], update[4], update[5];
      _estimate = Sophus::SE3d::exp(u) * _estimate;
    }

  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out) const override {}
};


class EdgeICP: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeICP(const Eigen::Vector3d& X2): _X2(X2) {}

    virtual void computeError() override {
      const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
      Sophus::SE3d Rt = v->estimate();
      _error = measurement() - Rt * _X2;
    }

    virtual void linearizeOplus() override {
      // fill _jacobianOPlusXi;
      const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
      Sophus::SE3d Rt = v->estimate();
      Eigen::Vector3d X2_transf = Rt * _X2;
      _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
      _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(X2_transf);
    }

    // Auto-diff is not working well. No idea why
    // template <class T>
    // bool operator() (const T* params, T* errors) const {
    //   const Eigen::Map<const Sophus::SE3<T>> Rt_sophus(params);
    //   Eigen::Matrix<T, 3, 1> X2(T(_X2.x()), T(_X2.y()), T(_X2.z()));
    //   Eigen::Matrix<T, 3, 1> X2_transf = Rt_sophus * X2;
    //   errors[0] = T(_measurement[0]) - X2_transf.x();
    //   errors[1] = T(_measurement[1]) - X2_transf.y();
    //   errors[2] = T(_measurement[2]) - X2_transf.z();
    //   return true;
    // }
    // G2O_MAKE_AUTO_AD_FUNCTIONS;



    virtual bool read(std::istream&) override {}
    virtual bool write(std::ostream&) const override {}

  private:
    Eigen::Vector3d _X2;
};


// Solve ICP with non-linear optimization
void bundleAdjustment(
  const vector<Eigen::Vector3d> &pts1,
  const vector<Eigen::Vector3d> &pts2,
  Eigen::Matrix3d &R, Eigen::Vector3d &t) {

    Sophus::SE3d pose_init(R, t);

    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(pose_init);
    optimizer.addVertex(vertex_pose);

    for (int i = 0; i < pts1.size(); ++i)
    {
      EdgeICP *edge = new EdgeICP(pts2[i]);
      edge->setVertex(0, vertex_pose);
      edge->setMeasurement(pts1[i]);
      edge->setInformation(Eigen::Matrix3d::Identity());
      optimizer.addEdge(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(50);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization ICP costs time: " << time_used.count() << " seconds." << endl;
    
    Sophus::SE3d optim_pose = vertex_pose->estimate();
    R = optim_pose.so3().unit_quaternion().toRotationMatrix();
    t = optim_pose.translation();
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

