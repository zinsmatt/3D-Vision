#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/EXTERNAL/ceres/autodiff.h"
#include <g2o/core/auto_differentiation.h>

#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <ceres/rotation.h>
#include <Eigen/Dense>

using namespace Sophus;
using namespace Eigen;
using namespace std;



class VertexCamera: public g2o::BaseVertex<9, Eigen::Matrix<double, 9, 1>>      //  here the camera is parameterized as a 9-vector (angle axis, t, f, k1, k2) 
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Matrix<double, 9, 1>::Zero();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Map<const Eigen::Matrix<double, 9, 1>>(update);
    }

    virtual bool read(std::istream&) override { return true; }
    virtual bool write(std::ostream&) const override { return true; }
};

class VertexLandmark: public g2o::BaseVertex<3, Eigen::Vector3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d::Zero();       
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Map<const Eigen::Vector3d>(update);
    }

    virtual bool read(std::istream&) override { return true; }
    virtual bool write(std::ostream&) const override { return true; }
};


class EdgeReprojection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera, VertexLandmark>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection() { }

    template <class T>
    bool operator() (const T* camera, const T* landmark, T* residuals) const 
    {
        T Xc[3];
        ceres::AngleAxisRotatePoint(camera, landmark, Xc);
        Xc[0] += camera[3];
        Xc[1] += camera[4];
        Xc[2] += camera[5];

        T Xp[2];
        Xp[0] = Xc[0] / Xc[2];
        Xp[1] = Xc[1] / Xc[2];

        T n2 = Xp[0] * Xp[0] + Xp[1] * Xp[1];
        T r = T(1.0) + n2 * (camera[7] + n2 * camera[8]);

        T uv[2];
        uv[0] = -Xp[0] * camera[6] * r;
        uv[1] = -Xp[1] * camera[6] * r;

        residuals[0] = T(_measurement[0]) - uv[0];
        residuals[1] = T(_measurement[1]) - uv[1];
        return true;
    }

    virtual bool read(std::istream&) override { return true; }
    virtual bool write(std::ostream&) const override { return true; }
    
    G2O_MAKE_AUTO_AD_FUNCTIONS  // use autodiff

};


int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
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


    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    auto* cameras = dataset.mutable_cameras();
    std::vector<VertexCamera*> camera_vertices;
    for (int i = 0; i < dataset.num_cameras(); ++i)
    {
        auto *c = new VertexCamera();
        c->setId(i);
        c->setEstimate(Eigen::Map<Eigen::Matrix<double, 9, 1>>(cameras + (i*dataset.camera_block_size())));
        optimizer.addVertex(c);
        camera_vertices.push_back(c);
    }
    
    auto* landmarks = dataset.mutable_points();
    std::vector<VertexLandmark*> landmark_vertices;
    for (int i = 0; i < dataset.num_points(); ++i)
    {
        auto* l = new VertexLandmark();
        l->setId(dataset.num_cameras() + i);
        l->setEstimate(Eigen::Map<Eigen::Vector3d>(landmarks + i*dataset.point_block_size()));
        l->setMarginalized(true);
        optimizer.addVertex(l);
        landmark_vertices.push_back(l);
    }

    auto* observations = dataset.observations();
    auto* cam_indices = dataset.camera_index();
    auto* landmark_indices = dataset.point_index();
    for (int i = 0; i < dataset.num_observations(); ++i)
    {
        auto* e = new EdgeReprojection();
        e->setVertex(0, camera_vertices[cam_indices[i]]);
        e->setVertex(1, landmark_vertices[landmark_indices[i]]);
        e->setMeasurement(Eigen::Map<const Eigen::Vector2d>(observations + i*2));
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);


    for (int i = 0; i < dataset.num_cameras(); ++i)
    {
        Eigen::Matrix<double, 9, 1> in = camera_vertices[i]->estimate();
        double *out = cameras + i * dataset.camera_block_size();
        for (int i = 0; i < 9; ++i)
        {
            out[i] = in[i];
        }
    }
    for (int i = 0; i < dataset.num_points(); ++i)
    {
        Eigen::Vector3d X = landmark_vertices[i]->estimate();
        landmarks[i*3] = X.x();
        landmarks[i*3+1] = X.y();
        landmarks[i*3+2] = X.z();
    }

    dataset.WriteToPLYFile("after_ba_g2o.ply");

    return 0;
}
