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

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

  // Optimize
  int max_iter = 10000;
  for (int it = 0; it < max_iter; ++it)
  {
    // std::cout << "iter " << it << " : ";
    // std::cout << "abc = " << a << " " << b << " " << c << "\n";
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d e = Eigen::Vector3d::Zero();
    double total_err = 0.0;
    for (int i = 0; i < N; ++i)
    {
      double x = x_data[i];
      double gx = g(x, a, b, c);
      double err = y_data[i] - gx;
      total_err += err * err;

      // Compute derivative of f(x) (F(x) = 1/2 sum(||f(x)||^2)
      Eigen::Vector3d J(0.0, 0.0, 0.0);
      J[0] = -x * x * gx; // df/da
      J[1] = -x * gx;  // df/db
      J[2] = -gx;   // df/dc

      // left part of the normal equation
      H += J * J.transpose();
      // right part
      e += -J * err;
    }
    // std::cout << "H = " << H << "\n";
    // std::cout << "total error: " << total_err << "\n";

    // solve Hx = e
    Eigen::Vector3d delta_x = H.inverse() * e;
    // Eigen::Vector3d delta_x = H.ldlt().solve(e); // second version

    // std::cout << "delta_x = " << delta_x.transpose() << "\n";

    if (delta_x.norm() < 1e-4)
      break;

    a += delta_x[0];
    b += delta_x[1];
    c += delta_x[2];
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
  
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
