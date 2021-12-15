#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
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
        Rt = g2o::SE3Quat::exp(Eigen::Map<Eigen::Matrix<double, 6, 1>>(data));

        f = data[6];
        k1 = data[7];
        k2 = data[8];
    }

    void set_to(double* data) const
    {
        Eigen::Matrix<double, 6, 1> rt = Rt.log();
        
        for (int i = 0; i < 6; ++i)
            data[i] = rt[i];
        

        data[6] = f;
        data[7] = k1;
        data[8] = k2;
    }

    g2o::SE3Quat Rt; // g2o::SE3Quat is stored as rotation first and then translation
    double f = 0.0, k1 = 0.0, k2 = 0.0;
};

class VertexCamera: public g2o::BaseVertex<6, Camera>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        
        _estimate = Camera();
        // _estimate = g2o::SE3Quat();
    }

    virtual void oplusImpl(const double *update) override {
        // _estimate = g2o::SE3Quat::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(update)) * _estimate;
        _estimate.Rt = g2o::SE3Quat::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(update)) * _estimate.Rt;

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


class EdgeReprojection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexLandmark, VertexCamera>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection(double f, double k1, double k2) : f_(f), k1_(k1), k2_(k2) { }

    virtual void computeError() override {
        const VertexCamera* v_cam = static_cast<VertexCamera*>(_vertices[1]);
        const VertexLandmark* v_point = static_cast<VertexLandmark*>(_vertices[0]);
        auto cam = v_cam->estimate();
        Eigen::Vector3d X = v_point->estimate();
        Eigen::Vector3d X_cam = cam.Rt * X;
        X_cam /= X_cam.z();

        // auto p2 = X_cam.x() * X_cam.x() + X_cam.y() * X_cam.y();
        // auto r = 1.0 + p2 * (k1_ + (p2 * k2_));

        Eigen::Vector2d uv = -X_cam.head<2>() *  f_ ; // minus because of the dataset projection
        _error = _measurement - uv;
    }


    virtual void linearizeOplus() override {
        const VertexCamera *v_cam = static_cast<VertexCamera*>(_vertices[1]);
        const VertexLandmark *v_landmark = static_cast<VertexLandmark*>(_vertices[0]);
        Eigen::Vector3d X = v_landmark->estimate();
        auto cam = v_cam->estimate();
        Eigen::Vector3d Xc = cam.Rt * X;

        double x = Xc.x();
        double y = Xc.y();
        double z = Xc.z();
        double z_2 = z * z;
        
        Eigen::Matrix<double, 2, 3> dedXc;
        dedXc << -f_ / z, 0.0, f_ * x / z_2,   // -f_ is because of the dataset projection function (not standard)
                0.0, -f_ / z, f_ * y / z_2;
        Eigen::Matrix3d dXcdR;
        dXcdR << 0.0, -z, y,
                 z, 0.0, -x,
                 -y, x, 0.0;
        Eigen::Matrix3d dXcdt = Eigen::Matrix3d::Identity();
        _jacobianOplusXj.block<2, 3>(0, 0) = dedXc * -dXcdR; // -dXcdR <=> -Xc^
        _jacobianOplusXj.block<2, 3>(0, 3) = dedXc * dXcdt;
        
        _jacobianOplusXi = dedXc * cam.Rt.rotation().matrix();

         _jacobianOplusXi *= -1; // take the negative jacobian because the error is defined as (measurement - projection)
         _jacobianOplusXj *= -1;


    }

    virtual bool read(std::istream&) override {}
    virtual bool write(std::ostream&) const override {}
    private:
        double f_, k1_, k2_;

};


int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem dataset(argv[1]);
    dataset.Normalize();
    // dataset.Perturb(0.1, 0.5, 0.5);
    dataset.WriteToPLYFile("initial_pc.ply");

    std::cout << "\n";
    std::cout << "nb cameras: " << dataset.num_cameras() << std::endl;
    std::cout << "nb landmarks: " << dataset.num_points() << std::endl;
    std::cout << "nb observations: " << dataset.num_observations() << std::endl;
    std::cout << "nb parameters: " << dataset.num_parameters() << std::endl;
    std::cout << "check: " << dataset.num_cameras() * 9 + dataset.num_points()*3 << std::endl;


    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
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
        auto c = Camera(cameras + (cam_indices[i]*dataset.camera_block_size()));
        auto* e = new EdgeReprojection(c.f, c.k1, c.k2);
        e->setVertex(1, camera_vertices[cam_indices[i]]);
        e->setVertex(0, landmark_vertices[landmark_indices[i]]);
        e->setMeasurement(Eigen::Map<const Eigen::Vector2d>(observations + i*2));
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);


    for (int i = 0; i < dataset.num_cameras(); ++i)
    {
        camera_vertices[i]->estimate().set_to(cameras + (i * dataset.camera_block_size()));
        // auto v = camera_vertices[i]->estimate().log();
        // double* pt = cameras + (i * dataset.camera_block_size());
        // for (int j = 0; j < 6; ++j)
        // {
        //     pt[j] = v[j];
        // }
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
