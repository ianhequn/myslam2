#include <gtest/gtest.h>
#include "myslam/g2o_types.h"

using namespace myslam;

TEST(PoseGraph, CorrectLoopReducesDriftAndKeepsAnchorFixed) {
    using BlockSolver = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolver>())));

    std::vector<VertexPose*> vertices;
    for (int i = 0; i < 11; ++i) {
        auto* vertex = new VertexPose;
        vertex->setId(i);
        vertex->setEstimate(SE3(SO3(), Vec3(i * 1.1, 0, 0)));
        vertex->setFixed(i == 0);
        ASSERT_TRUE(optimizer.addVertex(vertex));
        vertices.push_back(vertex);
    }
    for (int i = 1; i < 11; ++i) {
        auto* edge = new EdgePoseGraph;
        edge->setVertex(0, vertices[i - 1]);
        edge->setVertex(1, vertices[i]);
        edge->setMeasurement(SE3(SO3(), Vec3(1.1, 0, 0)));
        edge->setInformation(Mat66::Identity());
        ASSERT_TRUE(optimizer.addEdge(edge));
    }
    auto* loop = new EdgePoseGraph;
    loop->setVertex(0, vertices[0]);
    loop->setVertex(1, vertices[10]);
    loop->setMeasurement(SE3(SO3(), Vec3(10, 0, 0)));
    loop->setInformation(Mat66::Identity() * 100.0);
    auto* kernel = new g2o::RobustKernelHuber;
    kernel->setDelta(1.0);
    loop->setRobustKernel(kernel);
    ASSERT_TRUE(optimizer.addEdge(loop));
    ASSERT_TRUE(optimizer.initializeOptimization());
    ASSERT_GT(optimizer.optimize(30), 0);
    EXPECT_NEAR(vertices[0]->estimate().translation().x(), 0.0, 1e-9);
    EXPECT_NEAR(vertices[10]->estimate().translation().x(), 10.0, 0.05);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
