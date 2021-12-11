#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <Eigen/Dense>

using namespace Sophus;
using namespace Eigen;
using namespace std;

struct Camera
{
    Camera() {}

    Camera(double* data)
    {
        R = Sophus::SO3d::exp(Eigen::Vector3d(data[0], data[1], data[2]));
        t = Eigen::Vector3d(data[3], data[4], data[5]);
        f = data[6];
        k1 = data[7];
        k2 = data[8];
    }

    void set_to(double* data) const
    {
        Eigen::Vector3d r = R.log();
        for (int i = 0; i < 3; ++i)
        {
            data[i] = r[i];
            data[i+3] = t[i];
        }
        data[6] = f;
        data[7] = k1;
        data[8] = k2;
    }

    Sophus::SO3d R;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double f = 0.0, k1 = 0.0, k2 = 0.0;
};

class VertexCamera: public g2o::BaseVertex<9, Camera>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Camera();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.R = Sophus::SO3d::exp(Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.R;
        _estimate.t += Eigen::Map<const Eigen::Vector3d>(update+3);
        _estimate.f += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    virtual bool read(std::istream&) override {}
    virtual bool write(std::ostream&) const override {}
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

    virtual bool read(std::istream&) override {}
    virtual bool write(std::ostream&) const override {}
};


class EdgeReprojection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera, VertexLandmark>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection() { }

    virtual void computeError() override {
        const VertexCamera* v_cam = static_cast<VertexCamera*>(_vertices[0]);
        const VertexLandmark* v_point = static_cast<VertexLandmark*>(_vertices[1]);
        auto cam = v_cam->estimate();
        Eigen::Vector3d X = v_point->estimate();
        Eigen::Vector3d X_cam = (cam.R * X) + cam.t;
        X_cam /= -X_cam.z(); // minus because of dataset
        auto p2 = X_cam.x() * X_cam.x() + X_cam.y() * X_cam.y();
        auto r = 1.0 + p2 * (cam.k1 + (p2 * cam.k2));

        Eigen::Vector2d uv = X_cam.head<2>() * r * cam.f;
        _error = _measurement - uv;
    }

    virtual bool read(std::istream&) override {}
    virtual bool write(std::ostream&) const override {}

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
        c->setEstimate(Camera(cameras + (i*dataset.camera_block_size())));
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
        camera_vertices[i]->estimate().set_to(cameras + (i * dataset.camera_block_size()));
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
