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
  std::cout << "Steepest-Descent \n";

  double aa = 1.0, bb = 2.0, cc = 1.0;
  double a = 2.0, b = -1.0, c = -5.0;

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
    double total_err = 0.0;
    for (int i = 0; i < N; ++i)
    {
      double err = y_data[i] - g(x_data[i], a, b, c);
      total_err += err*err;

      // Compute derivative of F(x) = 0.5 * sum(f(x)^2) = 0.5 * sum((y-g(x))^2)
      J[0] += -x_data[i] * x_data[i] * g(x_data[i], a, b, c) * err;
      J[1] += -x_data[i] * g(x_data[i], a, b, c) * err;
      J[2] += -g(x_data[i], a, b, c) * err;
    }

    std::cout << "J = " << J.transpose() << "\n";
    std::cout << "total error: " << total_err << "\n";

    if (J.norm() < 0.01)
      break;

    // Very small step
    double S = 100000;
    a -= J[0] / S;
    b -= J[1] / S;
    c -= J[2] / S;
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
