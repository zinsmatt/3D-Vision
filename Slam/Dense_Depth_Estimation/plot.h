#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>


using namespace cv;
using namespace Eigen;


void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);
void plotCur(const Mat &cur);


void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);