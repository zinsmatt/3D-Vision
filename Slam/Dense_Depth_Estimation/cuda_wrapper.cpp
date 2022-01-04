#include <iostream>
#include <vector>
#include <fstream>
#include <thread>

#include "kernels.cuh"
#include "cuda_wrapper.h"
#include "constants.h"


void update_cuda(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, cv::Mat depth, cv::Mat cov2)
{
    Sophus::SE3d Trc = Tcr.inverse();
    double Tcr_data[3][4];
    double Trc_data[3][4];

    Eigen::Matrix<double, 3, 4> temp_cr = Tcr.matrix3x4();
    Eigen::Matrix<double, 3, 4> temp_rc = Trc.matrix3x4();
    for (int i = 0; i < 3; ++i)
    {

        for (int j = 0; j < 4; ++j)
        {
            Tcr_data[i][j] = temp_cr(i, j);
            Trc_data[i][j] = temp_rc(i, j);
        }
    }
    wrapper_update_cuda(ref.ptr<unsigned char>(0), cur.ptr<unsigned char>(0), Tcr_data, Trc_data, depth.ptr<double>(0), cov2.ptr<double>(0));
}
