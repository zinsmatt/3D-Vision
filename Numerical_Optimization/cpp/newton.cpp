#include <iostream>
#include <chrono>
#include <random>
#include <map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "matplotlibcpp.h"


using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

double g(double x, double a, double b, double c)
{
  // return a*x*x+b*x+c;
  return std::exp(a * x * x + b * x + c);
}

int main(int argc, char **argv) {
  std::cout << "Newton \n";

  double aa = 1.0, bb = 2.0, cc = 1.0;
  double a = 2.30, b = -1.20, c = 5.50;

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

  // Optimize
  int max_iter = 100000;
  for (int it = 0; it < max_iter; ++it)
  {
    std::cout << "iter " << it << " : ";
    std::cout << "abc = " << a << " " << b << " " << c << "\n";
    Eigen::Vector3d J(0.0, 0.0, 0.0);
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    double total_err = 0.0;
    for (int i = 0; i < N; ++i)
    {
      double x = x_data[i];
      double gx = g(x, a, b, c);
      double err = y_data[i] - gx;
      total_err += err * err;

      // Compute derivative of F(x) = 0.5 * sum(f(x)^2)
      J[0] += -x * x * gx * err;
      J[1] += -x * gx * err;
      J[2] += -gx * err;

      double fe = err * gx;
      double g2 = g(x, 2*a, 2*b, 2*c);
      double feg2 = fe-g2;

      H(0, 0) += -std::pow(x, 4) * feg2;
      H(1, 1) += -std::pow(x, 2) * feg2;
      H(2, 2) += -feg2;

      H(0, 1) += -std::pow(x, 3) * feg2;
      H(0, 2) += -std::pow(x, 2) * feg2;

      H(1, 0) += -std::pow(x, 3) * feg2;
      H(1, 2) += -x * feg2;

      H(2, 0) += -std::pow(x, 2) * feg2;
      H(2, 1) += -x * feg2;
    }

    std::cout << "J = " << J.transpose() << "\n";
    std::cout << "total error: " << total_err << "\n";

    Eigen::Vector3d delta_x = -H.inverse() * J;
    std::cout << "delta_x = " << delta_x.transpose() << "\n";

    if (delta_x.norm() < 0.0001)
      break;

    a += delta_x[0];
    b += delta_x[1];
    c += delta_x[2];
  }
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
