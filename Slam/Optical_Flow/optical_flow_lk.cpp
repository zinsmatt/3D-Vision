//
// Created by Xiang on 2017/12/19.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "../data/LK1.png";  // first image
string file_2 = "../data/LK2.png";  // second image

inline double get(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    double xx = x - floor(x);
    double yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}


void opticalFlowLK_linear(const cv::Mat& img1, const cv::Mat& img2, 
                          const std::vector<cv::Point2f>& kps1, std::vector<cv::Point2f>& kps2,
                          std::vector<bool>& success)
{
    kps2 = kps1;
    success.resize(kps1.size(), true);
    for (int iter = 0; iter < 10; ++iter)
    {
        for (int i = 0; i < kps1.size(); ++i)
        {
            auto& kp = kps1[i];
            int half_w_size = 4;

            int idx = 0;
            Eigen::Matrix<double, Eigen::Dynamic, 2> A(static_cast<int>(std::pow(half_w_size*2+1, 2)), 2);
            Eigen::Matrix<double, Eigen::Dynamic, 1> b(static_cast<int>(std::pow(half_w_size*2+1, 2)), 1);

            for (int xx = -half_w_size; xx <= half_w_size; ++xx)
            {
                for (int yy = -half_w_size; yy <= half_w_size; ++yy)
                {
                    double grad_x = 0.5 * (get(img1, kp.x+xx+1, kp.y+yy) - get(img1, kp.x+xx-1, kp.y+yy));
                    double grad_y = 0.5 * (get(img1, kp.x+xx, kp.y+yy+1) - get(img1, kp.x+xx, kp.y+yy-1));
                    double grad_t = get(img2, kps2[i].x+xx, kps2[i].y+yy) - get(img1, kp.x+xx, kp.y+yy);
                    A(idx, 0) = grad_x;
                    A(idx, 1) = grad_y;
                    b(idx, 0) = grad_t;
                    ++idx;
                }
            }
            Eigen::Vector2d uv = -(A.transpose() * A).inverse() * A.transpose() * b;

            if (std::isnan(uv.x()) || std::isnan(uv.y()))
            {
                success[i] = false;
            }

            kps2[i].x += uv.x();
            kps2[i].y += uv.y();
        }
    }
}

void opticalFlowLK_gauss_newton(const cv::Mat& img1, const cv::Mat& img2, 
                                const std::vector<cv::Point2f>& kps1, std::vector<cv::Point2f>& kps2,
                                std::vector<bool>& success, bool inverse=false, bool reinit_kps2=true)
{
    if (reinit_kps2)
    {
        std::cout << "re-init pts in img2 with points in img1" << std::endl;
        kps2 = kps1;
    }
    int max_iters = 10;
    success.resize(kps1.size(), true);
    for (int i = 0; i < kps1.size(); ++i)
    {
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d g = Eigen::Vector2d::Zero();
        double prev_cost = 0.0;
        double cost = 0.0;
        int half_w_size = 4;

        for (int iter = 0; iter < max_iters; ++iter)
        {
            prev_cost = cost;
            cost = 0.0;
            if (!inverse)
            {
                H = Eigen::Matrix2d::Zero();
                g = Eigen::Vector2d::Zero();
            }
            else
            {
                g = Eigen::Vector2d::Zero();
            }

            for (int xx = -half_w_size; xx <= half_w_size; ++xx)
            {
                for (int yy = -half_w_size; yy <= half_w_size; ++yy)
                {
                    double err = get(img1, kps1[i].x+xx, kps1[i].y+yy) - get(img2, kps2[i].x+xx, kps2[i].y+yy);
                    double grad_x, grad_y;
                    if (!inverse)
                    {
                        grad_x = 0.5 * (get(img2, kps2[i].x+xx+1, kps2[i].y+yy) -   get(img2, kps2[i].x+xx-1, kps2[i].y+yy));
                        grad_y = 0.5 * (get(img2, kps2[i].x+xx,   kps2[i].y+yy+1) - get(img2, kps2[i].x+xx,   kps2[i].y+yy-1));
                    }
                    else if (iter == 0) // if inverse mode, j and H are only computed once
                    {
                        grad_x = 0.5 * (get(img1, kps1[i].x+xx+1, kps1[i].y+yy) -   get(img1, kps1[i].x+xx-1, kps1[i].y+yy));
                        grad_y = 0.5 * (get(img1, kps1[i].x+xx,   kps1[i].y+yy+1) - get(img1, kps1[i].x+xx,   kps1[i].y+yy-1));
                    }
                    Eigen::Vector2d J(-grad_x, -grad_y);
                    if (inverse == false || iter == 0)
                    {
                        H += J * J.transpose();
                    }
                    g += -J * err;
                    cost += err * err;
                }
            }

            Eigen::Vector2d uv = H.ldlt().solve(g);
            if (std::isnan(uv.x()) || std::isnan(uv.y()))
            {
                success[i] = false;
                break;
            }

            if (iter > 0 && prev_cost < cost)
                break;

            kps2[i].x += uv.x();
            kps2[i].y += uv.y();

            if (uv.norm() < 1e-2)
                break; // converged
        }
    }

    
}

void opticalFlowLK_gauss_newton_pyramid(const cv::Mat& img1, const cv::Mat& img2, 
                                        const std::vector<cv::Point2f>& kps1, std::vector<cv::Point2f>& kps2,
                                        std::vector<bool>& success, int n_layers)
{
    int factor = 2;
    double scale = 1.0 / std::pow(factor, n_layers-1);
    std::vector<cv::Point2f> kps2_s;
    for (int si = 0; si < n_layers; ++si)
    {
        std::cout << "scale = " << scale << "\n";
        cv::Mat img1_s;
        cv::resize(img1, img1_s, cv::Size(), scale, scale);
        cv::Mat img2_s;
        cv::resize(img2, img2_s, cv::Size(), scale, scale);
        std::vector<cv::Point2f> kps1_s = kps1;
        for (auto& p : kps1_s)
        {
            p.x *= scale;
            p.y *= scale;
        }
        std::cout << "img size = " << img1_s.size() << "\n";
        opticalFlowLK_gauss_newton(img1_s, img2_s, kps1_s, kps2_s, success, true, si==0);

        if (si < n_layers-1)
        {
            for (auto& p : kps2_s)
            {
                p.x *= factor;
                p.y *= factor;
            }
        }

        scale *= factor;
    }
    kps2 = kps2_s;
}

int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

   
    // // use opencv's flow for validation
    // vector<Point2f> pt1, pt2;
    // for (auto &kp: kp1) pt1.push_back(kp.pt);
    // vector<uchar> status;
    // vector<float> error;
    // auto t1 = chrono::steady_clock::now();
    // cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    // auto t2 = chrono::steady_clock::now();
    // auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "optical flow by opencv: " << time_used.count() << endl;


    std::vector<cv::Point2f> pts1(kp1.size()), pts2;
    for (int i = 0; i < kp1.size(); ++i)
        pts1[i] = kp1[i].pt;
    std::vector<uchar> status;
    std::vector<float> errors;
    auto t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, errors);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;


    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pts2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pts2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pts1[i], pts2[i], cv::Scalar(0, 250, 0));
        }
    }
    // cv::imshow("tracked by opencv", img2_CV);
    // cv::waitKey(0);
    cv::imwrite("lk_opencv.png", img2_CV);


    // --- LK with linear solution
    std::vector<cv::Point2f> pts3;
    std::vector<bool> success;
    t1 = chrono::steady_clock::now();
    opticalFlowLK_linear(img1, img2, pts1, pts3, success);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by LK linear: " << time_used.count() << endl;

    Mat img2_LK;
    cv::cvtColor(img2, img2_LK, CV_GRAY2BGR);
    for (int i = 0; i < pts3.size(); i++) {
        if (status[i]) {
            cv::circle(img2_LK, pts3[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_LK, pts1[i], pts3[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imwrite("lk_linear.png", img2_LK);


    // --- LK with Gauss-Newton
    std::vector<cv::Point2f> pts4;
    t1 = chrono::steady_clock::now();
    opticalFlowLK_gauss_newton(img1, img2, pts1, pts4, success, false);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by LK Gauss-Newton: " << time_used.count() << endl;
    Mat img2_GN;
    cv::cvtColor(img2, img2_GN, CV_GRAY2BGR);
    for (int i = 0; i < pts4.size(); i++) {
        if (status[i]) {
            cv::circle(img2_GN, pts4[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_GN, pts1[i], pts4[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imwrite("lk_gauss-newton.png", img2_GN);

    // --- LK with Gauss-Newton (inverse)
    std::vector<cv::Point2f> pts5;
    t1 = chrono::steady_clock::now();
    opticalFlowLK_gauss_newton(img1, img2, pts1, pts5, success, true);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by LK Gauss-Newton inverse: " << time_used.count() << endl;

    Mat img2_GN_inv;
    cv::cvtColor(img2, img2_GN_inv, CV_GRAY2BGR);
    for (int i = 0; i < pts5.size(); i++) {
        if (status[i]) {
            cv::circle(img2_GN_inv, pts5[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_GN_inv, pts1[i], pts5[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imwrite("lk_gauss-newton_inverse.png", img2_GN_inv);


    // --- LK with Pyramidal Gauss-Newton (multi-layers)
    std::vector<cv::Point2f> pts6;
    t1 = chrono::steady_clock::now();
    opticalFlowLK_gauss_newton_pyramid(img1, img2, pts1, pts6, success, 4);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by LK Pyramidal Gauss-Newton: " << time_used.count() << endl;

    Mat img2_GN_Py;
    cv::cvtColor(img2, img2_GN_Py, CV_GRAY2BGR);
    for (int i = 0; i < pts6.size(); i++) {
        if (status[i]) {
            cv::circle(img2_GN_Py, pts6[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_GN_Py, pts1[i], pts6[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imwrite("lk_gauss-newton_pyramidal.png", img2_GN_Py);


    return 0;
}
