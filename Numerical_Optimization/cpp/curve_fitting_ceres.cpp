#include <iostream>
#include <chrono>
#include <random>
#include <map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "matplotlibcpp.h"


using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;


template <class T>
T g(T x, T a, T b, T c)
{
  return ceres::exp(a * x * x + b * x + c);
}



struct CurveFittingCost {

  CurveFittingCost(double x, double y) : _x(x), _y(y) {}

  template <class T>
  bool operator() (const T* const abc, T *residuals) const
  {
    residuals[0] = T(_y) - g(T(_x), abc[0], abc[1], abc[2]);
    return true;
  }

  private:
  const double _x, _y;
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


  // Add the residual blocks
  double abc[3] = {a, b, c}; // parameters
  ceres::Problem problem;
  for (int i = 0; i < N; ++i)
  {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(
        new CurveFittingCost(x_data[i], y_data[i])
        ),
      nullptr,
      abc
    );
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;



  std::cout << summary.BriefReport() << std::endl;
  std::cout << "Estimated: " << abc[0] << " "  << abc[1] << " "  << abc[2] << "\n";



  vector<double> final_y_data(N);
  for (int i = 0; i < N; ++i)
  {
    final_y_data[i] = g(x_data[i], abc[0], abc[1], abc[2]);
  }

  plt::scatter(x_data, y_data);
  std::map<std::string, std::string> parameters;
  parameters["c"] = "red";
  plt::scatter(x_data, final_y_data, 1.0, parameters);
  plt::show();  

  return 0;
}
