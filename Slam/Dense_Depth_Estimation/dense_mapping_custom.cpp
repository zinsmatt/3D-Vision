#include <iostream>
#include <vector>
#include <fstream>
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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "plot.h"

using namespace cv;


/**
 * Dataset from:
 * 
 *   http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
 * 
 * */


// ------------------------------------------------------------------
// parameters
const int boarder = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);
const double min_cov = 0.1;
const double max_cov = 10;
const double epsilon = 1e-10;
// ------------------------------------------------------------------



inline double getBilinearInterpolatedValue_eigen(const Mat &img, const Eigen::Vector2d &pt) {
    uchar *d = &img.data[int(pt[1]) * img.step + int(pt[0])];
    double xx = pt[0] - floor(pt[0]);
    double yy = pt[1] - floor(pt[1]);
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}



// ------------------------------------------------------------------

inline Vector3d px2cam(const Vector2d& px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

inline Vector2d cam2px(const Vector3d& p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}


bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


double ZNCC(const cv::Mat& im1, const Eigen::Vector2d& pt1, const cv::Mat& im2, Eigen::Vector2d& pt2)
{
    // no need to consider block partly outside because of boarder
    // std::vector<double> v1(ncc_area, 0.0), v2(ncc_area, 0.0); // much slower
    double v1[ncc_area], v2[ncc_area];
    double s1 = 0.0, s2 = 0.0;
    int idx = 0;
    for (int i = -ncc_window_size; i <= ncc_window_size; ++i)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; ++j)
        {
            double val_1 = static_cast<double>(im1.at<uchar>(pt1.y()+i, pt1.x()+j)) / 255;
            Eigen::Vector2d temp_p2 = pt2;
            temp_p2[0] += j;
            temp_p2[1] += i;
            double val_2 = getBilinearInterpolatedValue_eigen(im2, temp_p2);

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
    auto zncc =  numerator / (std::sqrt(den1 * den2 + epsilon));
    // std::cout << "zncc = " << zncc << "\n";
    return zncc;
}

bool epipolar_search(const cv::Mat& ref, const cv::Mat& cur, const Sophus::SE3d& Tcr, const Eigen::Vector2d& pt, double depth_mu, double depth_sigma2, Eigen::Vector2d& best_pc, Eigen::Vector2d& epipolar_dir)
{
    double depth_sigma = std::sqrt(depth_sigma2);
    double dmax = depth_mu + 3 * depth_sigma;
    double dmin = depth_mu - 3 * depth_sigma;
    dmin = std::max(0.1, dmin);

    Eigen::Vector3d pn((pt.x()-cx) / fx, (pt.y() - cy) / fy, 1.0);
    pn.normalize();
    Eigen::Vector3d P_max = pn * dmax;
    Eigen::Vector3d P_min = pn * dmin;
    Eigen::Vector3d P_mu = pn * depth_mu;


    Eigen::Vector2d pc_max = cam2px(Tcr * P_max);
    Eigen::Vector2d pc_min = cam2px(Tcr * P_min);
    Eigen::Vector2d pc_mu = cam2px(Tcr * P_mu);

    Eigen::Vector2d epipolar_line = pc_max - pc_min;
    epipolar_dir = epipolar_line.normalized();

    double step = 0.7;
    int nb_samples = std::ceil(epipolar_line.norm() / step);


    double half_range = 0.5 * epipolar_line.norm();
    if (half_range > 100) half_range = 100;

    Eigen::Vector2d p = pc_min;
    double best_zncc = -1.0;
    best_pc = pc_mu;


    // for (int i = 0; i < nb_samples; ++i)
    for (double l = -half_range; l<= half_range; l+= 0.7)
    {
        Eigen::Vector2d p = pc_mu + l * epipolar_dir;

        if (p.x() < boarder || p.x() >= width-boarder || p.y() < boarder || p.y() >= height-boarder)
            continue; // p is outside the cur image

        double zncc = ZNCC(ref, pt, cur, p);
        if (zncc > best_zncc)
        {
            best_zncc = zncc;
            best_pc = p;
        }

        // p += epipolar_dir * step;
    }

    // std::cout << best_zncc << "\n";
    if (best_zncc < 0.85)
        return false;
    else
        return true;
}

void update_depth_filter(const Eigen::Vector2d& pr, const Eigen::Vector2d& pc, const Sophus::SE3d& Tcr, const Eigen::Vector2d& epipolar_dir, cv::Mat& depth, cv::Mat& cov2)
{
    Sophus::SE3d Trc = Tcr.inverse();

    Eigen::Vector3d fr = px2cam(pr);
    fr.normalize();
    Eigen::Vector3d fc = px2cam(pc);
    fc.normalize();
    Eigen::Vector3d f2 = Trc.so3() * fc;
    Eigen::Vector3d trc = Trc.translation();


    // Solve the system of equation for triangulating depth
    Eigen::Matrix2d A;
    Eigen::Vector2d b;
    A(0, 0) = fr.dot(fr);
    A(0, 1) = -fr.dot(f2);
    A(1, 0) = f2.dot(fr);
    A(1, 1) = -f2.dot(f2);
    b[0] = fr.dot(trc);
    b[1] = f2.dot(trc);
    Eigen::Vector2d res = A.inverse() * b;
    Eigen::Vector3d P1 = fr * res[0];
    Eigen::Vector3d P2 = trc + fc * res[1];
    Eigen::Vector3d P_est = (P1 + P2) * 0.5;
    double depth_obs = P_est.norm(); //depth obs

    // Estimate depth uncertainty 
    Eigen::Vector3d P = fr * depth_obs;
    Eigen::Vector3d a = P - trc;
    Eigen::Vector3d t = trc.normalized();
    double alpha = std::acos(fr.dot(t));
    double beta = std::acos(a.normalized().dot(-t));
    Eigen::Vector2d pc2 = pc + epipolar_dir;
    Eigen::Vector3d fc2 = px2cam(pc2);
    fc2.normalize();
    double beta_2 = std::acos(fc2.dot(-t));
    double gamma = M_PI - alpha - beta_2;
    double d_noise = trc.norm() * std::sin(beta_2) / std::sin(gamma); // sinus law
    double sigma_obs = depth_obs - d_noise;
    double sigma2_obs = sigma_obs * sigma_obs; // sigma2 obs

    // Depth fusion
    double d = depth.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x()));
    double sigma2 = cov2.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x()));

    double d_fused = (sigma2_obs * d + sigma2 * depth_obs) / (sigma2 + sigma2_obs);
    double sigma2_fused = (sigma2 * sigma2_obs) / (sigma2 + sigma2_obs);

    depth.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x())) = d_fused;
    cov2.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x())) = sigma2_fused;
}


void update(const cv::Mat& ref, const cv::Mat& cur, const Sophus::SE3d& Tcr, cv::Mat &depth, cv::Mat &cov2)
{
    Eigen::Vector2d pc;
    Eigen::Vector2d epipolar_dir;
    for (int j = boarder; j < width-boarder; ++j)
    {
        for (int i = boarder; i < height-boarder; ++i)
        {
            double depth_mu = depth.at<double>(i, j);
            double depth_sigma2 = cov2.at<double>(i, j);
            if (depth_sigma2 < min_cov || depth_sigma2 > max_cov) 
                continue;
            Eigen::Vector2d pr(j, i);
            bool found = epipolar_search(ref, cur, Tcr, pr, depth_mu, depth_sigma2, pc, epipolar_dir);
            if (!found)
                continue;

            // showEpipolarMatch(ref, cur, pr, pc);

            update_depth_filter(pr, pc, Tcr, epipolar_dir, depth, cov2);
        }
    }
    // std::cout << depth << "\n";

}


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
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        std::cout << "Time used: " << time_used.count() << "s\n";
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        plotCur(curr);
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
