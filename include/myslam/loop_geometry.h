#pragma once

#include "myslam/common_include.h"

namespace myslam {

// Check that a historical landmark still projects near its ORB observation.
inline bool LoopReprojectionConsistent(const Vec3& point_camera,
                                       const Mat33& intrinsics,
                                       const cv::Point2f& observation,
                                       double max_error_px) {
    if (!point_camera.allFinite() || point_camera.z() <= 1e-6 ||
        !std::isfinite(observation.x) || !std::isfinite(observation.y) ||
        !std::isfinite(max_error_px) || max_error_px <= 0) return false;
    const Vec3 projected = intrinsics * point_camera;
    if (!projected.allFinite() || projected.z() <= 1e-6) return false;
    const double dx = projected.x() / projected.z() - observation.x;
    const double dy = projected.y() / projected.z() - observation.y;
    return dx * dx + dy * dy <= max_error_px * max_error_px;
}

}  // namespace myslam
