// Replay a saved graph without reprocessing images.
#include "myslam/g2o_types.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace myslam;

class RightUpdatePose : public VertexPose {
   public:
    void oplusImpl(const double* update) override {
        Eigen::Map<const Vec6> delta(update);
        _estimate = _estimate * SE3::exp(delta);
    }
};

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: replay_pose_graph before.txt loops.txt output.txt "
                     "loop_weight huber_delta iterations [lm|lm_right|gn] [max_translation_m]\n";
        return 1;
    }
    double weight, delta, max_translation = 0.0;
    int iterations;
    try {
        weight = std::stod(argv[4]);
        delta = std::stod(argv[5]);
        iterations = std::stoi(argv[6]);
        if (argc == 9) max_translation = std::stod(argv[8]);
    } catch (const std::exception&) {
        std::cerr << "Invalid optimizer parameters\n";
        return 1;
    }
    if (!(weight > 0) || !(delta > 0) || !std::isfinite(weight) ||
        !std::isfinite(delta) || !std::isfinite(max_translation) ||
        max_translation < 0 || iterations <= 0) return 1;

    std::ifstream poses(argv[1]), loops(argv[2]);
    if (!poses || !loops) {
        std::cerr << "Cannot open input files\n";
        return 1;
    }
    using BlockSolver = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;
    g2o::SparseOptimizer optimizer;
    const std::string method = argc >= 8 ? argv[7] : "lm";
    if (method == "lm" || method == "lm_right") {
        optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>())));
    } else if (method == "gn") {
        optimizer.setAlgorithm(new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>())));
    } else {
        std::cerr << "Unknown optimizer method\n";
        return 1;
    }
    std::vector<VertexPose*> vertices;
    std::vector<unsigned long> frame_ids;
    std::string line;
    while (std::getline(poses, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        unsigned long id;
        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
        if (!(in >> id)) return 1;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                if (!(in >> matrix(r, c)) || !std::isfinite(matrix(r, c))) return 1;
        VertexPose* vertex = method == "lm_right"
                                 ? static_cast<VertexPose*>(new RightUpdatePose)
                                 : new VertexPose;
        vertex->setId(static_cast<int>(vertices.size()));
        vertex->setFixed(vertices.empty());
        const Eigen::Matrix3d rotation = matrix.block<3, 3>(0, 0);
        if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-5 ||
            std::abs(rotation.determinant() - 1.0) > 1e-5) {
            std::cerr << "Invalid rotation matrix in keyframe trajectory\n";
            return 1;
        }
        Eigen::Quaterniond rotation_q(rotation);
        rotation_q.normalize();
        SE3 twc(SO3(rotation_q), matrix.block<3, 1>(0, 3));
        vertex->setEstimate(twc.inverse());
        if (!optimizer.addVertex(vertex)) return 1;
        frame_ids.push_back(id);
        vertices.push_back(vertex);
    }
    if (vertices.size() < 2) return 1;
    for (size_t i = 1; i < vertices.size(); ++i) {
        auto* edge = new EdgePoseGraph;
        edge->setVertex(0, vertices[i - 1]);
        edge->setVertex(1, vertices[i]);
        edge->setMeasurement(vertices[i]->estimate() * vertices[i - 1]->estimate().inverse());
        edge->setInformation(Mat66::Identity());
        if (!optimizer.addEdge(edge)) return 1;
    }
    int loop_count = 0, filtered_count = 0;
    while (std::getline(loops, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        size_t a, b;
        int inliers;
        double tx, ty, tz, qx, qy, qz, qw;
        if (!(in >> a >> b >> inliers >> tx >> ty >> tz >> qx >> qy >> qz >> qw) ||
            a >= vertices.size() || b >= vertices.size() || a == b) return 1;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        if (!q.coeffs().allFinite() || q.norm() < 1e-8 ||
            !std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz)) return 1;
        if (max_translation > 0 && Vec3(tx, ty, tz).norm() > max_translation) {
            ++filtered_count;
            continue;
        }
        q.normalize();
        auto* edge = new EdgePoseGraph;
        edge->setVertex(0, vertices[a]);
        edge->setVertex(1, vertices[b]);
        edge->setMeasurement(SE3(SO3(q), Vec3(tx, ty, tz)));
        edge->setInformation(Mat66::Identity() * weight);
        auto* kernel = new g2o::RobustKernelHuber;
        kernel->setDelta(delta);
        edge->setRobustKernel(kernel);
        if (!optimizer.addEdge(edge)) return 1;
        ++loop_count;
    }
    if (loop_count == 0 || !optimizer.initializeOptimization()) return 1;
    optimizer.computeActiveErrors();
    double before = optimizer.activeRobustChi2();
    int completed = optimizer.optimize(iterations);
    optimizer.computeActiveErrors();
    double after = optimizer.activeRobustChi2();
    std::cout << "keyframes=" << vertices.size() << " loop_edges=" << loop_count
              << " filtered=" << filtered_count
              << " iterations=" << completed << " robust_cost=" << before
              << " -> " << after << '\n';
    if (completed <= 0 || !std::isfinite(after) || after > before) return 1;
    std::ofstream out(argv[3]);
    if (!out) return 1;
    out << std::fixed << std::setprecision(9);
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto matrix = vertices[i]->estimate().inverse().matrix();
        out << frame_ids[i];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) out << ' ' << matrix(r, c);
        out << '\n';
    }
    return out ? 0 : 1;
}
