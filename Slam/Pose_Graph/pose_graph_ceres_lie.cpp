#include <iostream>
#include <fstream>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>



using namespace std;

//----------------------------------------------------------------
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
//----------------------------------------------------------------


class PoseSE3LieAlgebra
{
    public:
    PoseSE3LieAlgebra(const Sophus::SE3d& T12, const Eigen::Matrix<double, 6, 6>& information) : T12_(T12) {
        sqrt_information_ = Eigen::Matrix<double, 6, 6>::Identity();
        for (int i = 0; i < 6; ++i)
        {
            sqrt_information_(i, i) = std::sqrt(information(i, i));
        }
    }

    bool operator() (const double* const T1, const double* const T2,
                     double *residuals) const
    {
        Eigen::Map<const Sophus::SE3d> Rt1(T1);
        Eigen::Map<const Sophus::SE3d> Rt2(T2);

        Eigen::Matrix<double, 6, 1> res = (T12_.inverse() * Rt1.inverse() * Rt2).log();

        for (int i = 0; i < 6; ++i)
            residuals[i] = res[i] * sqrt_information_(i, i);

        return true;        
    }

    private:
        Sophus::SE3d T12_;
        Eigen::Matrix<double, 6, 6> sqrt_information_;
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

    ceres::Problem problem;

    std::unordered_map<int, std::vector<double>> poses;
    std::vector<Sophus::SE3d> constraints;
    std::vector<std::pair<int, int>> indices;
    std::vector<Eigen::Matrix<double, 6, 6>> informations;

    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            std::vector<double> pose(7);
            int index = 0;
            fin >> index;
            double data[7];
            for (int i = 0; i < 7; ++i)
            {
                fin >> data[i];
            }
            pose[0] = data[3];
            pose[1] = data[4];
            pose[2] = data[5];
            pose[3] = data[6];
            pose[4] = data[0];
            pose[5] = data[1];
            pose[6] = data[2];
            poses[index] = pose;
            
            problem.AddParameterBlock(poses[index].data(), 7, new LocalParameterizationSE3);
            if (index == 0)
                problem.SetParameterBlockConstant(poses[index].data());

        } else if (name == "EDGE_SE3:QUAT") {
            int idx1, idx2;
            fin >> idx1 >> idx2;
            double data[7];
            for (int i = 0; i < 7; ++i)
            {
                fin >> data[i];
            }

            Eigen::Matrix<double, 6, 6> information;
            for (int i = 0; i < 6; ++i)
            {
                for (int j = i; j < 6; ++j)
                {
                    fin >> information(i, j);
                    information(j, i) = information(i, j);
                }
            }
            informations.push_back(information);

            Eigen::Vector3d t(data);
            Eigen::Quaterniond q(data+3);


            Sophus::SE3d T12(q, t);
            constraints.push_back(T12);
            indices.push_back(std::make_pair(idx1, idx2));

            auto* functor = new PoseSE3LieAlgebra(T12, information);
            auto* cost_function = new ceres::NumericDiffCostFunction<PoseSE3LieAlgebra, ceres::CENTRAL, 6, 7, 7>(functor);
            problem.AddResidualBlock(cost_function, nullptr, &poses[idx1][0], &poses[idx2][0]);
        }
        if (!fin.good()) break;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::ofstream fout("result_lie_ceres.g2o");
    for (auto it : poses)
    {
        fout << "VERTEX_SE3:QUAT ";
        fout << it.first << " ";
        const auto& v = it.second;
        fout << v[4] << " " << v[5] << " " << v[6] << " " << v[7] << " " << v[0] << " " << v[1] << " " << v[2] << "\n"; 
    }

    for (int i = 0; i < constraints.size(); ++i)
    {
        fout << "EDGE_SE3:QUAT ";
        fout << indices[i].first << " " << indices[i].second << " ";
        const auto& Rt = constraints[i];
        auto t = Rt.translation();
        auto q = Rt.unit_quaternion();
        fout << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";
        for (int k = 0; k < 6; ++k)
            for (int j = k; j < 6; ++j)
                fout << informations[i](k, j) << " ";
        fout << "\n";
    }

    fout.close();
    return 0;
}

