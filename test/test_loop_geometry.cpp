#include <gtest/gtest.h>
#include <limits>
#include "myslam/loop_geometry.h"

using namespace myslam;

TEST(LoopGeometry, RequiresPositiveDepthAndConsistentProjection) {
    Mat33 k = Mat33::Identity();
    k(0, 0) = 100;
    k(1, 1) = 100;
    k(0, 2) = 50;
    k(1, 2) = 40;
    EXPECT_TRUE(LoopReprojectionConsistent(Vec3(1, 0, 10), k, cv::Point2f(61, 40), 2));
    EXPECT_FALSE(LoopReprojectionConsistent(Vec3(1, 0, 10), k, cv::Point2f(64, 40), 2));
    EXPECT_FALSE(LoopReprojectionConsistent(Vec3(1, 0, -10), k, cv::Point2f(40, 40), 2));
    EXPECT_FALSE(LoopReprojectionConsistent(Vec3(1, 0, 10), k, cv::Point2f(60, 40), 0));
    EXPECT_FALSE(LoopReprojectionConsistent(
        Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 10), k, cv::Point2f(60, 40), 2));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
