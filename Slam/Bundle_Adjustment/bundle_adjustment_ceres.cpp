#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include <ceres/rotation.h>
using namespace std;



struct ReprojectionError
{

    ReprojectionError(double x, double y) : x_(x), y_(y) {}


    template <class T>
    bool operator() (const T* const camera,
                     const T* const point,
                     T* residuals) const
    {
        // camera: angle-axis, translation, f, k1, k2
        // points: x, y, z
        T X_cam[3];
        ceres::AngleAxisRotatePoint(camera, point, X_cam);
        X_cam[0] += T(camera[3]);
        X_cam[1] += T(camera[4]);
        X_cam[2] += T(camera[5]);

        X_cam[0] /= -X_cam[2]; // minus because of the dataset convention
        X_cam[1] /= -X_cam[2];
        
        T p2 = X_cam[0] * X_cam[0] + X_cam[1] * X_cam[1];
        T r = T(1.0) + T(camera[7]) * p2 + T(camera[8]) * p2 * p2;

        X_cam[0] *= r;
        X_cam[1] *= r;

        X_cam[0] *= T(camera[6]);
        X_cam[1] *= T(camera[6]);

        residuals[0] = x_ - X_cam[0];
        residuals[1] = y_ - X_cam[1];
        return true;
    }

    static ceres::CostFunction* Create(double x, double y){
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(
            new ReprojectionError(x, y)
        );
    }

    double x_, y_;
};


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    BALProblem dataset(argv[1]);
    dataset.Normalize();
    dataset.Perturb(0.1, 0.5, 0.5);
    dataset.WriteToPLYFile("initial_pc.ply");

    std::cout << "\n";

    std::cout << "nb cameras: " << dataset.num_cameras() << std::endl;
    std::cout << "nb landmarks: " << dataset.num_points() << std::endl;
    std::cout << "nb observations: " << dataset.num_observations() << std::endl;
    std::cout << "nb parameters: " << dataset.num_parameters() << std::endl;
    std::cout << "check: " << dataset.num_cameras() * 9 + dataset.num_points()*3 << std::endl;

    ceres::Problem problem;

    const double* observations = dataset.observations();
    for (int i = 0; i < dataset.num_observations(); ++i)
    {
        auto* cost = ReprojectionError::Create(observations[i*2], observations[i*2+1]);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), dataset.mutable_camera_for_observation(i), dataset.mutable_point_for_observation(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 16;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";


    dataset.WriteToPLYFile("after_ba.ply");


   return 0;
}
