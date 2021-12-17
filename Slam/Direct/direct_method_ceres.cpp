#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <ceres/ceres.h>
#include <chrono>

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

Eigen::Vector3d get_3D_point_from_depth(const Eigen::Vector2d& p, double depth, const Eigen::Matrix3d& K)
{
    return Eigen::Vector3d(depth * (p.x() - K(0, 2)) / K(0, 0),
                           depth * (p.y() - K(1, 2)) / K(1, 1),
                           depth);
}

using namespace Sophus;
// Local parameterization needed to handle SE3 from Sophus (from Sophus/test/ceres/)
class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<SE3d const> const T(T_raw);
    Eigen::Map<Vector6d const> const delta(delta_raw);
    Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * SE3d::exp(delta);
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
        jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }


  virtual int GlobalSize() const { return SE3d::num_parameters; }

  virtual int LocalSize() const { return SE3d::DoF; }
};



struct PhotometricError: public ceres::SizedCostFunction<9, 7>
{
    PhotometricError(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Vector2d& p1, const Eigen::Vector3d& P1, const Eigen::Matrix3d& K, int half_w_size)
    : _img1(img1), _img2(img2), _p1(p1), _P1(P1), _K(K), _half_w_size(half_w_size) {}


    virtual bool Evaluate(double const* const *params,
                          double *residuals,
                          double **jacobians) const {
        const Eigen::Map<const Sophus::SE3d> Rt(params[0]);

        Eigen::Vector3d P2 = Rt * _P1;
        Eigen::Vector3d p2 = _K * P2;
        p2 /= p2.z();

        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = P2.x();
        double Y = P2.y();
        double Z = P2.z();
        double X2 = X * X;
        double Y2 = Y * Y;
        double Z2 = Z * Z;

        double x3 = _P1.x();
        double y3 = _P1.y();
        double z3 = _P1.z();
        double qx = params[0][0];
        double qy = params[0][1];
        double qz = params[0][2];
        double qw = params[0][3];


        if (P2.z() < 0) // invalid depth
        {
            // fill residuals and jacobians to 0
            for (int i = 0; i < 9; ++i)
            {
                residuals[i] = 0;
                if (jacobians!=nullptr && jacobians[0]!=nullptr)
                {
                    for (int j = 0; j < 7; ++j)
                    {
                        jacobians[0][i*7+j] = 0.0;
                    }
                }
            }
            return true;
        }

        if (p2.x() < _half_w_size || p2.x() > _img2.cols - _half_w_size 
            || p2.y() < _half_w_size || p2.y() > _img2.rows - _half_w_size)
        {
            // fill residuals and jacobians to 0
            for (int i = 0; i < 9; ++i)
            {
                residuals[i] = 0;
                if (jacobians!=nullptr && jacobians[0]!=nullptr)
                {
                    for (int j = 0; j < 7; ++j)
                    {
                        jacobians[0][i*7+j] = 0.0;
                    }
                }
            }
            return true;
        }

        int cnt = 0;
        for (int xx = -_half_w_size; xx <= _half_w_size; ++xx)
        {
            for (int yy = -_half_w_size; yy <= _half_w_size; ++yy)
            {

                double v1 = get(_img1, _p1.x() + xx, _p1.y() + yy);
                double v2 = get(_img2,  p2.x() + xx, p2.y() + yy);
                double err = v1 - v2;
                residuals[cnt] = err;

                if (jacobians && jacobians[0])
                {
                    double dx = 0.5 * (get(_img2, p2.x() + xx + 1, p2.y() + yy) - get(_img2, p2.x() + xx - 1, p2.y() + yy));
                    double dy = 0.5 * (get(_img2, p2.x() + xx, p2.y() + yy + 1) - get(_img2, p2.x() + xx, p2.y() + yy - 1));
                    Eigen::Vector2d dIdu(dx, dy);

                    Eigen::Matrix<double, 2, 3> dudXc;
                    dudXc << fx / Z, 0.0, -X * fx / Z2,
                            0.0, fy / Z, -Y * fy / Z2;
                    

                    Eigen::Matrix<double, 3, 4> dXcdq; // derivative of Xcam wrt. quaternions
                    dXcdq(0, 0) = 2*qy*y3 + 2*qz*z3;
                    dXcdq(0, 1) = 2*qw*z3 + 2*qx*y3 - 4*qy*x3;
                    dXcdq(0, 2) = -2*qw*y3 + 2*qx*z3 - 4*qz*x3;
                    dXcdq(0, 3) = 2*qy*z3 - 2*qz*y3;

                    dXcdq(1, 0) = -2*qw*z3 - 4*qx*y3 + 2*qy*x3;
                    dXcdq(1, 1) = 2*qx*x3 + 2*qz*z3;
                    dXcdq(1, 2) = 2*qw*x3 + 2*qy*z3 - 4*qz*y3;
                    dXcdq(1, 3) = -2*qx*z3 + 2*qz*x3;

                    dXcdq(2, 0) = 2*qw*y3 - 4*qx*z3 + 2*qz*x3;
                    dXcdq(2, 1) = -2*qw*x3 - 4*qy*z3 + 2*qz*y3;
                    dXcdq(2, 2) = 2*qx*x3 + 2*qy*y3;
                    dXcdq(2, 3) = 2*qx*y3 - 2*qy*x3;

                    Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J;
                    J.block<1, 4>(0, 0) = -dIdu.transpose() * dudXc * dXcdq;
                    J.block<1, 3>(0, 4) = -dIdu.transpose() * dudXc;

                    for (int i = 0; i < 7; ++i)
                    {
                        jacobians[0][cnt*7 + i] = J(0, i);
                    }
                }
                ++cnt;
            }
        }

        return true;
    }

    private:
        cv::Mat _img1, _img2;
        Eigen::Vector2d _p1;
        Eigen::Vector3d _P1;
        Eigen::Matrix3d _K;
        int _half_w_size;

};

// TO TEST with autodiff or numeric diff, but not easy to differentiate wrt. image (maybe try to combine Jets + image gradient with chain rule)
// struct PhotometricError
// {
//     PhotometricError(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Vector2d& p1, const Eigen::Vector3d& P1, const Eigen::Matrix3d& K, int half_w_size)
//     : _img1(img1), _img2(img2), _p1(p1), _P1(P1), _K(K), _half_w_size(half_w_size) {}

//     virtual bool operator() (const double* const params,
//                              double *residuals) const {
//         const Eigen::Map<const Sophus::SE3d> Rt(params);

//         Eigen::Vector3d P2 = Rt * _P1;
//         Eigen::Vector3d p2 = _K * P2;
//         p2 /= p2.z();

//         int j = 0;
//         for (int xx = -_half_w_size; xx <= _half_w_size; ++xx)
//         {
//             for (int yy = -_half_w_size; yy <= _half_w_size; ++yy)
//             {
//                 double v1 = get(_img1, _p1.x() + xx, _p1.y() + yy);
//                 double v2 = get(_img2, p2.x() + xx, p2.y() + yy);
//                 // debug << _p1.x() << " " << _p1.y() << " " << p2.x() << " " << p2.y() << "\n";
//                 debug << v1 << " " << v2 << "\n";
//                 double err = v1 - v2;
//                 residuals[j++] = err;
//             }
//         }

//         return true;
//     }


//     private:
//         cv::Mat _img1, _img2;
//         Eigen::Vector2d _p1;
//         Eigen::Vector3d _P1;
//         Eigen::Matrix3d _K;
//         int _half_w_size;
// };


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
    int nb_iters = 11;
    int half_w_size = 1;

    ceres::Problem problem;

    problem.AddParameterBlock(Rt.data(), 7, new LocalParameterizationSE3());
    for (int i = 0; i < px_ref.size(); ++i)
    {
        const auto& p1 = px_ref[i];
        Eigen::Vector3d P1 = get_3D_point_from_depth(p1, depth_ref[i], K);
        problem.AddResidualBlock(
            // new ceres::NumericDiffCostFunction<PhotometricError, ceres::CENTRAL, 9, 7>(
            //     new PhotometricError(img1, img2, p1, P1, K, half_w_size)
            // ),            
            new PhotometricError(img1, img2, p1, P1, K, half_w_size),
            nullptr,
            Rt.data()
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 11;

    ceres::Solver::Summary summary;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double >>(t2 - t1);
    cout << "optimization with ceres costs time: " << time_used.count() << " seconds." << endl;
    

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
    int nb_levels = 4;
    double factor = 0.5;

    std::vector<cv::Mat> pyr1, pyr2;
    std::vector<double> scales;
    for (int i = 0; i < nb_levels; ++i)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
            scales.push_back(1.0);
        }
        else
        {
            cv::Mat img1_r, img2_r;
            cv::resize(pyr1[i-1], img1_r, cv::Size(pyr1[i-1].cols * factor, pyr1[i-1].rows * factor));
            cv::resize(pyr2[i-1], img2_r, cv::Size(pyr2[i-1].cols * factor, pyr2[i-1].rows * factor));
            pyr1.push_back(img1_r);            
            pyr2.push_back(img2_r);            
            scales.push_back(scales[i-1] * factor);
        }
    }


    for (int l = nb_levels-1; l >= 0; l--)
    {

        cv::Mat img1_r = pyr1[l];
        cv::Mat img2_r = pyr2[l];
        double scale = scales[l];

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
    }
}


int main(int argc, char **argv) {

Sophus::SE3d r;
Eigen::Matrix<double, 6, 1> d;
d << 1, 2, 3, 0, 0, 0;
r =  Sophus::SE3d::exp(d) * r;
double* rr = r.data();
for (int i = 0; i<7;++i)
std::cout << rr[i] << " ";
cout << "\n\n";

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng(1994);
    int nPoints = 2000;
    int boarder = 40;
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
            Eigen::Vector3d P_ref = get_3D_point_from_depth(pixels_ref[i], depth_ref[i], K);
            Eigen::Vector3d uv = K * (Rt * P_ref);
            projections[i] = uv.hnormalized();
        }

        for (size_t i = 0; i < pixels_ref.size(); ++i) {
            auto p_ref = pixels_ref[i];
            auto p_cur = projections[i];
            if (p_cur[0] > 0 && p_cur[1] > 0 && p_cur[0] < img2_show.cols && p_cur[1] < img2_show.rows) {
                cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                        cv::Scalar(0, 250, 0));
            }
        }
        // cv::imshow("current", img2_show);
        // cv::waitKey();
        cv::imwrite("img_"+std::to_string(i) + "_ceres.png", img2_show);

    }
    // debug.close();
    return 0;
}
