#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <cmath>
#include <chrono>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


#include "constants.h"
#include "cuda_wrapper.h"
#include "kernels.cuh"
#include "plot.h"

using namespace cv;


/**
 * Dataset from:
 * 
 *   http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
 * 
 * */


inline double getBilinearInterpolatedValue_no_eigen(const unsigned char *img, double pt[2]) {
    const unsigned char* d = &img[(int)pt[1] * width + (int)pt[0]];
    double xx = pt[0] - floor(pt[0]);
    double yy = pt[1] - floor(pt[1]);
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[width]) +
            xx * yy * double(d[width + 1])) / 255.0;
}


inline void pix2cam_no_eigen(const double in[2], double out[3]) {
    out[0] = (in[0] - cx) / fx;
    out[1] =  (in[1] - cy) / fy;
    out[2] = 1.0;
}


inline void cam2pix_no_eigen(const double in[3], double out[2]) {
    out[0] = in[0] * fx / in[2] + cx;
    out[1] = in[1] * fy / in[2] + cy;
}


inline double norm3_no_eigen(const double in[3])
{
    return sqrt(in[0]*in[0] + in[1]*in[1] + in[2]*in[2]);
}


inline double norm2_no_eigen(const double in[2])
{
    return sqrt(in[0]*in[0] + in[1]*in[1]);
}


// inplace normalization vec 3

inline void normalize3_no_eigen(double in_out[3]) {
    double d = sqrt(in_out[0]*in_out[0] 
                    + in_out[1]*in_out[1]
                    + in_out[2]*in_out[2]);
    in_out[0] /= d;
    in_out[1] /= d;
    in_out[2] /= d;
}

// inplace normalization vec 2

inline void normalize2_no_eigen(double in_out[2]) {
    double d = sqrt(in_out[0]*in_out[0] + in_out[1]*in_out[1]);
    in_out[0] /= d;
    in_out[1] /= d;
}


inline void transform_no_eigen(double x[3], const double T[12], double out[3])
{
    for (int i = 0; i < 3; ++i)
    {
        out[i] = x[0] * T[i*4] + x[1] * T[i*4+1] + x[2] * T[i*4+2] +  T[i*4+3];
    }
}



double ZNCC_no_eigen(const unsigned char *im1, const double pt1[2], const unsigned char *im2, const double pt2[2])
{
    // no need to consider block partly outside because of boarder
    double v1[ncc_area], v2[ncc_area];
    double s1 = 0.0, s2 = 0.0;
    int idx = 0;
    for (int i = -ncc_window_size; i <= ncc_window_size; ++i)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; ++j)
        {
            double val_1 = ((double) im1[((int)pt1[1] + i) * width + (int)pt1[0] + j]) / 255;
            double temp_p2[2] = {pt2[0] + j, pt2[1] + i};
            double val_2 = getBilinearInterpolatedValue_no_eigen(im2, temp_p2);
            s1 += val_1;
            s2 += val_2;
            v1[idx] = val_1;
            v2[idx] = val_2;
            ++idx;
        }
    }

    double mean_1 = s1 / ncc_area;
    double mean_2 = s2 / ncc_area;

    double numerator = 0.0;
    double den1 = 0.0, den2 = 0.0;
    for (int i = 0; i < ncc_area; ++i)
    {
        double zv1 = v1[i] - mean_1;
        double zv2 = v2[i] - mean_2;
        numerator += zv1*zv2;
        den1 += zv1 * zv1;
        den2 += zv2 * zv2;
    }
    auto zncc =  numerator / (sqrt(den1 * den2 + epsilon));
    // std::cout << "zncc = " << zncc << "\n";
    return zncc;
}


bool epipolar_search_no_eigen(const unsigned char* ref, const unsigned char* cur, 
                          const double Tcr[12], const double pt[2],
                          double depth_mu, double depth_sigma2, 
                          double best_pc[2], double epipolar_dir[2])
{

    double depth_sigma = sqrt(depth_sigma2);
    double dmax = depth_mu + 3 * depth_sigma;
    double dmin = depth_mu - 3 * depth_sigma;
    dmin = max(0.1, dmin);

    double pn[3];
    pix2cam_no_eigen(pt, pn);
    normalize3_no_eigen(pn);
    double P_max[3] = {pn[0] * dmax, pn[1] * dmax, pn[2] * dmax};
    double P_min[3] = {pn[0] * dmin, pn[1] * dmin, pn[2] * dmin};
    double P_mu[3] = {pn[0] * depth_mu, pn[1] * depth_mu, pn[2] * depth_mu};

    double P_max_cur[3], P_min_cur[3], P_mu_cur[3];
    transform_no_eigen(P_max, Tcr, P_max_cur);
    transform_no_eigen(P_min, Tcr, P_min_cur);
    transform_no_eigen(P_mu, Tcr, P_mu_cur);


    double pc_max[2], pc_min[2], pc_mu[2];
    cam2pix_no_eigen(P_max_cur, pc_max);
    cam2pix_no_eigen(P_min_cur, pc_min);
    cam2pix_no_eigen(P_mu_cur, pc_mu);


    double epipolar_line[2] = {pc_max[0] - pc_min[0], pc_max[1] - pc_min[1]};
    epipolar_dir[0] = epipolar_line[0];
    epipolar_dir[1] = epipolar_line[1];
    normalize2_no_eigen(epipolar_dir);
    double epipolar_line_norm = norm2_no_eigen(epipolar_line);

    // double step = 0.7;
    // int nb_samples = std::ceil(epipolar_line.norm() / step);

    double half_range = 0.5 * epipolar_line_norm;
    if (half_range > 100) half_range = 100;

    double best_zncc = -1.0;
    for (double l = -half_range; l<= half_range; l+= 0.7)
    {
        double p[2] = {pc_mu[0] + l * epipolar_dir[0], pc_mu[1] + l * epipolar_dir[1]};

        if (p[0] < boarder || p[0] >= width-boarder || p[1] < boarder || p[1] >= height-boarder)
            continue; // p is outside the cur image

        double zncc = ZNCC_no_eigen(ref, pt, cur, p);
        if (zncc > best_zncc)
        {
            best_zncc = zncc;
            best_pc[0] = p[0];
            best_pc[1] = p[1];
        }
    }
    if (best_zncc < 0.85)
        return false;
    else
        return true;
}


double dot3_no_eigen(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


double det2_no_eigen(const double A[2][2])
{
    return A[0][0] * A[1][1] - A[1][0] * A[0][1];
}


void solve_Axb2_no_eigen(const double A[2][2], const double b[2], double res[2])
{
    double det_inv = 1.0 / det2_no_eigen(A);
    double A_inv[2][2];
    A_inv[0][0] = det_inv * A[1][1];
    A_inv[0][1] = -det_inv * A[0][1];
    A_inv[1][0] = -det_inv * A[1][0];
    A_inv[1][1] = det_inv * A[0][0];

    res[0] = A_inv[0][0] * b[0] + A_inv[0][1] * b[1];
    res[1] = A_inv[1][0] * b[0] + A_inv[1][1] * b[1];
}


void update_depth_filter_no_eigen(const double pr[2], const double pc[2], const double Trc[12], const double epipolar_dir[2], double *depth, double *cov2)
{
    double fr[3];
    pix2cam_no_eigen(pr, fr);
    normalize3_no_eigen(fr);

    double fc[3];
    pix2cam_no_eigen(pc, fc);
    normalize3_no_eigen(fc);
    
    double f2[3] = {dot3_no_eigen(Trc, fc),
                    dot3_no_eigen(Trc+4, fc),
                    dot3_no_eigen(Trc+8, fc)};

    double trc[3] = {Trc[3], Trc[7], Trc[11]};
    double A[2][2];
    double b[2];

    A[0][0] = dot3_no_eigen(fr, fr);
    A[0][1] = dot3_no_eigen(fr, f2);
    A[1][0] = dot3_no_eigen(f2, fr);
    A[1][1] = dot3_no_eigen(f2, f2);
    A[0][1] *= -1;
    A[1][1] *= -1;
    
    b[0] = dot3_no_eigen(fr, trc);
    b[1] = dot3_no_eigen(f2, trc);

    if (abs(det2_no_eigen(A)) < 1e-20) // not invertible
        return;

    double res[2];
    solve_Axb2_no_eigen(A, b, res);
    double P1[3] = {fr[0] * res[0], fr[1] * res[0], fr[2] * res[0]};
    double P2[3] = {trc[0] + fc[0] * res[1], trc[1] + fc[1] * res[1], trc[2] + fc[2] * res[1]};
    double P_est[3] = {(P1[0] + P2[0]) * 0.5, 
                       (P1[1] + P2[1]) * 0.5, 
                       (P1[2] + P2[2]) * 0.5};
    double depth_obs = norm3_no_eigen(P_est);

    double P[3] = {fr[0] * depth_obs, fr[1] * depth_obs, fr[2] * depth_obs};
    double a[3] = {P[0] - trc[0], P[1] - trc[1], P[2] - trc[2]};

    double t[3] = {trc[0], trc[1], trc[2]};
    normalize3_no_eigen(t);

    double alpha = acos(dot3_no_eigen(fr, t));
    double beta = acos(-dot3_no_eigen(a, t) / norm3_no_eigen(a));

    double pc2[2] = {pc[0] + epipolar_dir[0], pc[1] + epipolar_dir[1]};
    double fc2[3];
    pix2cam_no_eigen(pc2, fc2);
    normalize3_no_eigen(fc2);
    double beta_2 = acos(-dot3_no_eigen(fc2, t));

    double gamma = M_PI - alpha - beta_2;
    double d_noise = norm3_no_eigen(trc) * sin(beta_2) / sin(gamma); // sinus law
    double sigma_obs = depth_obs - d_noise;
    double sigma2_obs = sigma_obs * sigma_obs;


    // Depth fusion
    double d = depth[(int)pr[1] * width + (int)pr[0]];
    double sigma2 = cov2[(int)pr[1] * width + (int)pr[0]];

    double d_fused = (sigma2_obs * d + sigma2 * depth_obs) / (sigma2 + sigma2_obs);
    double sigma2_fused = (sigma2 * sigma2_obs) / (sigma2 + sigma2_obs);

    depth[(int)pr[1] * width + (int)pr[0]] = d_fused;
    cov2[(int)pr[1] * width + (int)pr[0]] = sigma2_fused;

}



void update_no_eigen(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, cv::Mat depth, cv::Mat cov2)
{
    Eigen::Vector2d pc;
    Eigen::Vector2d epipolar_dir;
    double pc_out[3];
    double epipolar_dir_out[2];

    Sophus::SE3d Trc = Tcr.inverse();
    double Tcr_data[12];
    double Trc_data[12];

    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Tcr_matrix = Tcr.matrix3x4();
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Trc_matrix = Trc.matrix3x4();
   
    int total=0;
    for (int j = boarder; j < width-boarder; ++j)
    {
        for (int i = boarder; i < height-boarder; ++i)
        {
            double depth_mu = depth.at<double>(i, j);
            double depth_sigma2 = cov2.at<double>(i, j);
            if (depth_sigma2 < min_cov || depth_sigma2 > max_cov)
                continue;
            Eigen::Vector2d pr(j, i);
            bool found = epipolar_search_no_eigen(ref.ptr<unsigned char>(0), cur.ptr<unsigned char>(0),
                                                  Tcr_matrix.data(), pr.data(),
                                                  depth_mu, depth_sigma2,
                                                  pc.data(), epipolar_dir.data());
            total += found;
            if (!found)
                continue;
            // showEpipolarMatch(ref, cur, pr, pc);

            update_depth_filter_no_eigen(pr.data(), pc.data(), Trc_matrix.data(), epipolar_dir.data(), depth.ptr<double>(0), cov2.ptr<double>(0));
        }
    }
    std::cout << "total found " << total << " / " << width*height << "\n";
}



bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);



int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // Read dataset
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // Initial depth image
    Mat ref = imread(color_image_files[0], 0); // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    Mat depth(height, width, CV_64F, init_depth);
    Mat depth_cov2(height, width, CV_64F, init_cov2);

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // T_C_W * T_W_R = T_C_R
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        update_no_eigen(ref, curr, pose_T_C_R, depth, depth_cov2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        std::cout << "Time used: " << time_used.count() << "s\n";

        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        plotCur(curr);
        // imshow("image", curr);
        // waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}



void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;
    double ave_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}
