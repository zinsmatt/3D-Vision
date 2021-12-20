#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;


class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update_v(update);
        _estimate = Sophus::SE3d::exp(update_v) * _estimate;
    }


    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        _estimate = Sophus::SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])
        );
        return true;
    }

    virtual bool write(ostream &os) const override {
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }
};

class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError() override {
        auto* v0 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        auto* v1 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[1]);
        _error = (_measurement.inverse() * v0->estimate().inverse() * v1->estimate()).log();
    }

    virtual void linearizeOplus() override {
        // auto* v0 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        auto* v1 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[1]);
        // auto T0 = v0->estimate();
        auto T1 = v1->estimate();

        Sophus::SE3d e = Sophus::SE3d::exp(_error);
        Eigen::Matrix<double, 6, 6> J_inv = Eigen::Matrix<double, 6, 6>::Zero();
        J_inv.block<3, 3>(0, 0) = Sophus::SO3d::hat(e.so3().log());
        J_inv.block<3, 3>(3, 3) = Sophus::SO3d::hat(e.so3().log());
        J_inv.block<3, 3>(0, 3) = Sophus::SO3d::hat(e.translation());
        J_inv *= 0.5;
        J_inv += Eigen::Matrix<double, 6, 6>::Identity();


        // also possible to approximate J_inv with Identity

        _jacobianOplusXi = -J_inv * T1.inverse().Adj();
        _jacobianOplusXj = J_inv * T1.inverse().Adj();
    }

    virtual bool read(std::istream& is) override {
        double data[7];
        for (int i = 0; i < 7; ++i)
            is >> data[i];
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        Eigen::Vector3d t(data);
        _measurement = Sophus::SE3d(q, t);
        for (int i = 0; i < _information.rows() && is.good(); ++i)
        {
            for (int j = i; j < _information.cols() && is.good(); ++j)
            {
                is >> _information(i, j);
                _information(j, i) = _information(i, j);
            }
        }
        return true;
    }

    virtual bool write(std::ostream& os) const override {
        auto *v0 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        auto *v1 = dynamic_cast<VertexSE3LieAlgebra*>(_vertices[1]);
        os << v0->id() << " " << v1->id() << " ";
        auto q = _measurement.unit_quaternion();
        auto t = _measurement.translation();
        os << t.x() << " " << t.y() << " " << t.z() << " ";
        os << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";

        for (int i = 0; i < _information.rows(); ++i)
        {
            for (int j = i; j < _information.cols(); ++j)
            {
                os << _information(i, j) << " ";
            }
        }
        os << std::endl;
        return true;
    }

};

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int vertexCnt = 0, edgeCnt = 0;

    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;

    ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v:vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
