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

    g2o::SE3Quat Rt; // g2o::SE3Quat stores rotation first and then translation
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
        _estimate.Rt = g2o::SE3Quat::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(update)) * _estimate.Rt;
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

        auto p2 = X_cam.x() * X_cam.x() + X_cam.y() * X_cam.y();
        auto r = 1.0 + p2 * (cam.k1 + (p2 * cam.k2));

        Eigen::Vector2d uv = -X_cam.head<2>() *  cam.f * r; // minus because of the dataset projection
        _error = _measurement - uv;
    }


    virtual void linearizeOplus()  override  {
        const VertexCamera *v_cam = static_cast<VertexCamera*>(_vertices[1]);
        const VertexLandmark *v_landmark = static_cast<VertexLandmark*>(_vertices[0]);
        Eigen::Vector3d X3d = v_landmark->estimate();
        auto cam = v_cam->estimate();
        Eigen::Vector3d Xc = cam.Rt * X3d;

        double X = Xc.x();
        double Y = Xc.y();
        double Z = Xc.z();
        double Z_2 = Z * Z;
        double f = cam.f;
        double k1 = cam.k1;
        double k2 = cam.k2;

        double x = X / Z;
        double y = Y / Z;
        double x2 = x*x;
        double y2 = y*y;
        double n2 = x*x + y*y;
        double n4 = n2 * n2;
        double r = 1.0 + n2 * k1 + n4 * k2;


        Eigen::Matrix2d dedxd = -Eigen::Matrix2d::Identity() * f; // minus because of the dataset projection

        Eigen::Matrix2d dxddxp = Eigen::Matrix2d::Identity();

        dxddxp(0, 0) = (2*k1*x+4*k2*x2*x+4*y2*x*k2)*x + r;
        dxddxp(0, 1) = x*(2*k1*y+4*x2*k2*y+4*k2*y2*y);
        dxddxp(1, 0) = y*(2*k1*x+4*k2*x2*x+4*y2*k2*x);
        dxddxp(1, 1) = (2*k1*y+4*x2*y*k2+4*k2*y2*y)*y + r;


        Eigen::Matrix<double, 2, 3> dxpdXc;
        dxpdXc << 1.0/Z, 0.0, -X / (Z_2),
                  0.0, 1.0/Z, -Y / (Z_2);


        Eigen::Matrix<double, 2, 3> dedXc = dedxd * dxddxp * dxpdXc;
        
        Eigen::Matrix3d dXcdR;
        dXcdR << 0.0, -Z, Y,
                 Z, 0.0, -X,
                 -Y, X, 0.0;

        Eigen::Matrix3d dXcdt = Eigen::Matrix3d::Identity();

        Eigen::Vector2d dedf(-r*x, -r*y);   // minus because of the dataset projection 
        Eigen::Vector2d dxddk1(x * n2, y * n2);
        Eigen::Vector2d dxddk2(x * n4, y * n4);
        Eigen::Vector2d dedk1 = dedxd * dxddk1;
        Eigen::Vector2d dedk2 = dedxd * dxddk2;

        _jacobianOplusXj.block<2, 3>(0, 0) = dedXc * -dXcdR; // -dXcdR <=> -Xc^
        _jacobianOplusXj.block<2, 3>(0, 3) = dedXc * dXcdt;
        _jacobianOplusXj.col(6) = dedf;
        _jacobianOplusXj.col(7) = dedk1;
        _jacobianOplusXj.col(8) = dedk2;
        
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
