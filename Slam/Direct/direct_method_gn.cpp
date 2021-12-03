#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>


using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

// baseline
double baseline = 0.573;
// paths
string left_file = "../data/left.png";
string disparity_file = "../data/disparity.png";
boost::format fmt_others("../data/%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


// bilinear interpolation
inline float get(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

Eigen::Vector3d get_3D_point_from_depth(const Eigen::Vector2d& p, double depth)
{
    return Eigen::Vector3d(depth * (p.x() - cx) / fx,
                           depth * (p.y() - cy) / fy,
                           depth);
}


/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const Eigen::Matrix3d& K,
    Sophus::SE3d &Rt // points from cam1 reference frame to cam2
)
{
    int nb_iters = 10;
    int half_w_size = 2;
    double prev_cost = 0.0;



    for (int iter = 0; iter < nb_iters; iter++)
    {
        std::cout << "Iter: " << iter << " ";
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        double total_cost = 0.0;
        int cnt_good = 0;
        for (int k = 0; k < px_ref.size(); ++k)
        {
            const auto& p1 = px_ref[k];
            Eigen::Vector3d P_ref = get_3D_point_from_depth(p1, depth_ref[k]);
            Eigen::Vector3d P2  = Rt * P_ref;
            double X2 = std::pow(P2.x(), 2);
            double Y2 = std::pow(P2.y(), 2);
            double Z2 = std::pow(P2.z(), 2);

            if (P2.z() < 0) // invalid depth
                continue;

            Eigen::Vector3d p2 = K  * P2;
            p2 /= p2.z();

            if (p2.x() < half_w_size || p2.x() > img2.cols - half_w_size 
                || p2.y() < half_w_size || p2.y() > img2.rows - half_w_size)
                continue;
            
            cnt_good++;
            for (int xx = -half_w_size; xx <= half_w_size; ++xx)
            {
                for (int yy = -half_w_size; yy <= half_w_size; ++yy)
                {
                    auto v1 = get(img1, p1.x() + xx, p1.y() + yy);
                    auto v2 = get(img2, p2.x() + xx, p2.y() + yy);
                    double dx = 0.5 * (get(img2, p2.x() + xx + 1, p2.y() + yy) - get(img2, p2.x() + xx - 1, p2.y() + yy));
                    double dy = 0.5 * (get(img2, p2.x() + xx, p2.y() + yy + 1) - get(img2, p2.x() + xx, p2.y() + yy - 1));
                    Eigen::Vector2d dIdu(dx, dy);
                    Eigen::Matrix<double, 2, 6> dudRt;
                    dudRt << fx/P2.z(), 0.0, -fx*P2.x() / Z2,-fx * P2.x() * P2.y() / Z2, fx + fx * X2 / Z2, -fx * P2.y() / P2.z(),
                             0.0, fy / P2.z(), -fy*P2.y()/Z2, -fy-fy * Y2 / Z2, fy * P2.x() * P2.y() / Z2, fy * P2.x() / P2.z();
                    Eigen::Matrix<double, 6, 1> J = -(dIdu.transpose() * dudRt).transpose();
                    double err = v1 - v2;
                    H += J * J.transpose();
                    g += -J.transpose() * err;
                    total_cost += err * err;
                }
            }
        }
        Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(g);
        std::cout << std::setw(4) << std::setprecision(3) << "\tcost: " << total_cost << "\t update norm: " << delta.norm() << "\n";

        if (std::isnan(delta[0]))
        {
            std::cout << "Error during optimization (linear equation solving failed)" << std::endl;
            break;
        }

        if (iter >  0 && total_cost > prev_cost)
        {
            std::cout << "Cost increased. Stop." << std::endl;
            break;
        }

        Rt = Sophus::SE3d::exp(delta) * Rt;
        prev_cost = total_cost;

        if (delta.norm() < 1e-3)
        {
            std::cout << "Optimization converged." << std::endl;
            break;
        }
    }
    std::cout << "translation: " << Rt.translation().transpose() << "\n";
    std::cout << "rotation: " << Rt.so3().unit_quaternion().toRotationMatrix() << "\n";

}


void DirectPoseEstimationPyramidal(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const Eigen::Matrix3d& K,
    Sophus::SE3d &Rt)
{
    int nb_levels = 1;
    double factor = 2.0;
    double scale = 1.0 / std::pow(factor, nb_levels-1);


    for (int l = 0; l < nb_levels; ++l)
    {
        cv::Mat img1_r, img2_r;
        cv::resize(img1, img1_r, cv::Size(), scale, scale);
        cv::resize(img2, img2_r, cv::Size(), scale, scale);

        Eigen::Matrix3d K_r = K;
        K_r(0, 0) *= scale;
        K_r(1, 1) *= scale;
        K_r(0, 2) *= scale;
        K_r(1, 2) *= scale;
        auto p_r = px_ref;
        for (auto& p : p_r)
        {
            p *= scale;
        }

        DirectPoseEstimationSingleLayer(img1_r, img2_r, p_r, depth_ref, K_r, Rt);

        scale *= factor;
    }
}


int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng(1994);
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;


    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d Rt;
    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, K, Rt);
        DirectPoseEstimationPyramidal(left_img, img, pixels_ref, depth_ref, K, Rt);


        // plot the projected pixels here
        cv::Mat img2_show;
        cv::cvtColor(img, img2_show, CV_GRAY2BGR);
        std::vector<Eigen::Vector2d> projections(pixels_ref.size());
        for (int i = 0; i < pixels_ref.size(); ++i)
        {
            Eigen::Vector3d P_ref = get_3D_point_from_depth(pixels_ref[i], depth_ref[i]);
            Eigen::Vector3d uv = K * (Rt * P_ref);
            projections[i] = uv.hnormalized();
        }

        for (size_t i = 0; i < pixels_ref.size(); ++i) {
            auto p_ref = pixels_ref[i];
            auto p_cur = projections[i];
            if (p_cur[0] > 0 && p_cur[1] > 0) {
                cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                        cv::Scalar(0, 250, 0));
            }
        }
        cv::imshow("current", img2_show);
        cv::waitKey();
        // cv::imwrite("img_"+std::to_string(i) + ".png", img2_show);

    }
    return 0;
}
