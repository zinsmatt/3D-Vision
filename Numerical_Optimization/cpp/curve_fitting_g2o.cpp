#include <iostream>
#include <chrono>
#include <random>
#include <map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

// #include "g2o/core/auto_differentiation.h"
// #include "g2o/core/base_unary_edge.h"
// #include "g2o/core/base_vertex.h"
// #include "g2o/core/optimization_algorithm_factory.h"
// #include "g2o/core/sparse_optimizer.h"
// #include "g2o/stuff/command_args.h"
// #include "g2o/stuff/sampler.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include "matplotlibcpp.h"

G2O_USE_OPTIMIZATION_LIBRARY(dense);


using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;


template <class T, class TT>
T g(TT x, T a, T b, T c)
{
  return ceres::exp(a * x * x + b * x + c);
}




class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d> {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override {
    _estimate << 0.0, 0.0, 0.0;
  }
  virtual void oplusImpl(const double* update) override {
    _estimate += Eigen::Vector3d(update);
  }

  virtual bool read(std::istream&) {}
  virtual bool write(std::ostream&) const {}
};

class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
  public:  
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CurveFittingEdge(double x): _x(x) {}

    // Without auto-diff
    virtual void computeError() override {
      const CurveFittingVertex* v = static_cast<CurveFittingVertex*>(_vertices[0]);
      const Eigen::Vector3d abc = v->estimate();
      _error(0, 0) = _measurement - g(_x, abc[0], abc[1], abc[2]);
    }

    // Analytical jacobian
    virtual void linearizeOplus() override {
      const CurveFittingVertex *v = static_cast<CurveFittingVertex*>(_vertices[0]);
      const Eigen::Vector3d abc = v->estimate();
      double y = g(_x, abc[0], abc[1], abc[2]);
      _jacobianOplusXi[0] = -_x * _x * y;
      _jacobianOplusXi[1] = -_x * y;
      _jacobianOplusXi[2] = -y;
    }



    // with auto-diff
    // template <class T>
    // bool operator() (const T* abc, T *error) const {
    //   error[0] = measurement() - g(_x, abc[0], abc[1], abc[2]);
    //   return true;
    // }

    virtual bool read(istream&) {}
    virtual bool write(ostream&) const {}

  private:
    double _x;


  // G2O_MAKE_AUTO_AD_FUNCTIONS  // use autodiff
};  


int main(int argc, char **argv) {
  std::cout << "Gauss-Newton \n";

  double aa = 1.0, bb = 2.0, cc = 1.0;
  double a = 2.0, b = -1.0, c = 5.5;

  double obs_sigma = 1.0;
  std::default_random_engine generator;
  std::normal_distribution<double> obs_noise_distrib(0.0, obs_sigma);


  int N = 100;
  std::vector<double> x_data(N), y_data(N);
  for (int i = 0; i < N; ++i)
  {
    x_data[i] = static_cast<double>(i) / 100.0;
    y_data[i] = g(x_data[i], aa, bb, cc) + obs_noise_distrib(generator);

  }


  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  // auto solver = new g2o::OptimizationAlgorithmGaussNewton(
  //   g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
  // );

  g2o::SparseOptimizer optimizer;
  // optimizer.setAlgorithm(solver);
  g2o::OptimizationAlgorithmProperty solverProperty;
  optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solverProperty));
  optimizer.setVerbose(true);


  // Create vertex
  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(a, b, c));
  v->setId(0);
  optimizer.addVertex(v);

  // Add edges
  for (int i = 0; i < N; ++i)
  {
    CurveFittingEdge * edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);
    edge->setMeasurement(y_data[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() / (obs_sigma * obs_sigma));
    optimizer.addEdge(edge);
  }


  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
  
  
  a = v->estimate()[0];
  b = v->estimate()[1];
  c = v->estimate()[2];

  std::cout << "final = " << a << " " << b << " " << c << "\n";



  vector<double> final_y_data(N);
  for (int i = 0; i < N; ++i)
  {
    final_y_data[i] = g(x_data[i], a, b, c);
  }

  plt::scatter(x_data, y_data);
  std::map<std::string, std::string> parameters;
  parameters["c"] = "red";
  plt::scatter(x_data, final_y_data, 1.0, parameters);
  plt::show();  


  return 0;
}
