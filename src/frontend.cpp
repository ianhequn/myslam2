//
//

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

#include <algorithm>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/loop_geometry.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

DEFINE_double(loop_association_px, 8.0, "Maximum ORB-to-feature association distance in pixels");
DEFINE_double(loop_pnp_reprojection_px, 8.0, "PnP RANSAC reprojection threshold in pixels");
DEFINE_double(loop_min_inlier_ratio, 0.25, "Minimum PnP inlier ratio");
DEFINE_double(loop_history_reprojection_px, 0.0,
              "Historical landmark reprojection threshold in pixels; 0 disables the check");

namespace myslam {

namespace {

int FindGoodORBMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                       std::vector<cv::DMatch>* good_matches = nullptr) {
    if (descriptors1.empty() || descriptors2.empty()) {
        return 0;
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    if (knn_matches.empty()) {
        return 0;
    }

    const float ratio_threshold = 0.75f;
    std::vector<cv::DMatch> local_good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() < 2) {
            continue;
        }

        const auto& best_match = match_pair[0];
        const auto& second_best_match = match_pair[1];
        if (best_match.distance <
            ratio_threshold * second_best_match.distance) {
            local_good_matches.push_back(best_match);
        }
    }

    if (good_matches) {
        *good_matches = local_good_matches;
    }

    return static_cast<int>(local_good_matches.size());
}

}  // namespace

Frontend::Frontend() {
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    orb_ = cv::ORB::create(1000);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;
//这里传入的参数ptr就是结构体的名字
    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());//Lastframe是智能指针 指向frameptr中的pose成员
    //只要上一帧存在就track
    }

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}
//下一个函数我们要插入关键帧
bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    ExtractORBForKeyframe();
    MatchORBWithLastKeyframe();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    DetectLoopCandidate();
    SetObservationsForKeyFrame();
    DetectFeatures();  // detect new features

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    AssociateORBWithMapPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

void Frontend::ExtractORBForKeyframe() {
    orb_->detectAndCompute(current_frame_->left_img_, cv::Mat(),
                           current_frame_->orb_keypoints_,
                           current_frame_->orb_descriptors_);
    current_frame_->orb_map_points_.clear();
    current_frame_->orb_map_points_.resize(current_frame_->orb_keypoints_.size());

    LOG(INFO) << "ORB keypoints in keyframe "
              << current_frame_->keyframe_id_ << ": "
              << current_frame_->orb_keypoints_.size();
}

void Frontend::AssociateORBWithMapPoints() {
    const double max_feature_distance = FLAGS_loop_association_px;
    if (!std::isfinite(max_feature_distance) || max_feature_distance <= 0) {
        LOG(ERROR) << "Invalid loop association threshold";
        return;
    }
    const double max_distance2 = max_feature_distance * max_feature_distance;
    int num_associated = 0;

    current_frame_->orb_map_points_.clear();
    current_frame_->orb_map_points_.resize(current_frame_->orb_keypoints_.size());

    for (size_t i = 0; i < current_frame_->orb_keypoints_.size(); ++i) {
        const cv::Point2f orb_pt = current_frame_->orb_keypoints_[i].pt;
        MapPoint::Ptr nearest_map_point = nullptr;
        double best_distance2 = max_distance2;

        for (const auto& feature : current_frame_->features_left_) {
            if (!feature || feature->is_outlier_) {
                continue;
            }

            auto map_point = feature->map_point_.lock();
            if (!map_point || map_point->is_outlier_) {
                continue;
            }

            const cv::Point2f diff = feature->position_.pt - orb_pt;
            const double distance2 = diff.x * diff.x + diff.y * diff.y;
            if (distance2 < best_distance2) {
                best_distance2 = distance2;
                nearest_map_point = map_point;
            }
        }

        if (nearest_map_point) {
            current_frame_->orb_map_points_[i] = nearest_map_point;
            num_associated++;
        }
    }

    LOG(INFO) << "Associated ORB keypoints with map points in keyframe "
              << current_frame_->keyframe_id_ << ": " << num_associated;
}

void Frontend::MatchORBWithLastKeyframe() {
    if (!last_orb_keyframe_) {
        last_orb_keyframe_ = current_frame_;
        return;
    }

    if (last_orb_keyframe_->orb_descriptors_.empty() ||
        current_frame_->orb_descriptors_.empty()) {
        LOG(WARNING) << "Skip ORB matching because descriptors are empty.";
        last_orb_keyframe_ = current_frame_;
        return;
    }

    std::vector<cv::DMatch> good_matches;
    const int num_good_matches =
        FindGoodORBMatches(last_orb_keyframe_->orb_descriptors_,
                           current_frame_->orb_descriptors_, &good_matches);

    LOG(INFO) << "ORB matches keyframe "
              << last_orb_keyframe_->keyframe_id_ << " -> "
              << current_frame_->keyframe_id_
              << ": good=" << num_good_matches;

    cv::Mat match_image;
    cv::drawMatches(last_orb_keyframe_->left_img_,
                    last_orb_keyframe_->orb_keypoints_,
                    current_frame_->left_img_,
                    current_frame_->orb_keypoints_, good_matches, match_image,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("./results/orb_match_latest.png", match_image);

    if (current_frame_->keyframe_id_ <= 5) {
        std::string filename =
            "./results/orb_match_kf_" +
            std::to_string(last_orb_keyframe_->keyframe_id_) + "_" +
            std::to_string(current_frame_->keyframe_id_) + ".png";
        cv::imwrite(filename, match_image);
    }

    last_orb_keyframe_ = current_frame_;
}

void Frontend::DetectLoopCandidate() {
    const int min_loop_keyframe_gap = 30;
    const int min_loop_good_matches = 80;

    if (!map_) {
        return;
    }

    if (current_frame_->orb_descriptors_.empty()) {
        LOG(WARNING) << "Skip loop candidate search because current ORB "
                        "descriptors are empty.";
        return;
    }

    auto all_keyframes = map_->GetAllKeyFrames();
    Frame::Ptr best_candidate = nullptr;
    std::vector<cv::DMatch> best_matches;
    int best_good_matches = 0;

    for (const auto& id_and_keyframe : all_keyframes) {
        Frame::Ptr history_keyframe = id_and_keyframe.second;
        if (!history_keyframe || history_keyframe == current_frame_) {
            continue;
        }

        if (current_frame_->keyframe_id_ <
            history_keyframe->keyframe_id_ + min_loop_keyframe_gap) {
            continue;
        }

        std::vector<cv::DMatch> good_matches;
        int num_good_matches =
            FindGoodORBMatches(history_keyframe->orb_descriptors_,
                               current_frame_->orb_descriptors_, &good_matches);

        if (num_good_matches > best_good_matches) {
            best_good_matches = num_good_matches;
            best_candidate = history_keyframe;
            best_matches = good_matches;
        }
    }

    if (!best_candidate) {
        LOG(INFO) << "Loop candidate search keyframe "
                  << current_frame_->keyframe_id_ << ": no history keyframe";
        return;
    }

    LOG(INFO) << "Loop candidate search keyframe "
              << current_frame_->keyframe_id_ << ": best keyframe "
              << best_candidate->keyframe_id_
              << ", good matches=" << best_good_matches;

    if (best_good_matches < min_loop_good_matches) {
        return;
    }

    int num_correspondences = 0;
    int num_inliers = 0;
    SE3 loop_relative_pose;
    if (!VerifyLoopCandidatePnP(best_candidate, best_matches,
                                &num_correspondences, &num_inliers,
                                &loop_relative_pose)) {
        LOG(INFO) << "Loop candidate rejected by PnP keyframe "
                  << best_candidate->keyframe_id_ << " -> "
                  << current_frame_->keyframe_id_
                  << ", good matches=" << best_good_matches
                  << ", correspondences=" << num_correspondences
                  << ", inliers=" << num_inliers;
        return;
    }

    LOG(INFO) << "Loop candidate accepted by PnP keyframe "
              << best_candidate->keyframe_id_ << " -> "
              << current_frame_->keyframe_id_
              << ", good matches=" << best_good_matches
              << ", correspondences=" << num_correspondences
              << ", inliers=" << num_inliers;

    map_->AddLoopConstraint(best_candidate->keyframe_id_,
                            current_frame_->keyframe_id_,
                            loop_relative_pose, num_inliers);

    cv::Mat loop_image;
    cv::drawMatches(best_candidate->left_img_, best_candidate->orb_keypoints_,
                    current_frame_->left_img_, current_frame_->orb_keypoints_,
                    best_matches, loop_image, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("./results/loop_candidate_latest.png", loop_image);

    std::string filename =
        "./results/loop_candidate_kf_" +
        std::to_string(best_candidate->keyframe_id_) + "_" +
        std::to_string(current_frame_->keyframe_id_) + ".png";
    cv::imwrite(filename, loop_image);
}

bool Frontend::VerifyLoopCandidatePnP(Frame::Ptr candidate,
                                      const std::vector<cv::DMatch>& matches,
                                      int* num_correspondences,
                                      int* num_inliers,
                                      SE3* relative_pose) {
    const int min_pnp_correspondences = 12;
    const int min_pnp_inliers = 20;
    const double min_inlier_ratio = FLAGS_loop_min_inlier_ratio;
    if (!std::isfinite(min_inlier_ratio) || min_inlier_ratio < 0 ||
        min_inlier_ratio > 1 || !std::isfinite(FLAGS_loop_pnp_reprojection_px) ||
        FLAGS_loop_pnp_reprojection_px <= 0 ||
        !std::isfinite(FLAGS_loop_history_reprojection_px) ||
        FLAGS_loop_history_reprojection_px < 0) {
        LOG(ERROR) << "Invalid loop verification thresholds";
        return false;
    }

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    std::set<unsigned long> used_landmark_ids;
    const SE3 candidate_pose = candidate->Pose();
    int rejected_history_reprojection = 0;

    for (const auto& match : matches) {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >=
                static_cast<int>(candidate->orb_keypoints_.size()) ||
            match.trainIdx >=
                static_cast<int>(current_frame_->orb_keypoints_.size())) {
            continue;
        }

        if (match.queryIdx >=
            static_cast<int>(candidate->orb_map_points_.size())) {
            continue;
        }

        auto map_point = candidate->orb_map_points_[match.queryIdx].lock();
        if (!map_point || map_point->is_outlier_) {
            continue;
        }

        if (used_landmark_ids.count(map_point->id_) > 0) {
            continue;
        }
        const Vec3 point_world = map_point->Pos();
        if (!point_world.allFinite()) continue;
        if (FLAGS_loop_history_reprojection_px > 0 &&
            !LoopReprojectionConsistent(candidate_pose * point_world,
                camera_left_->K(), candidate->orb_keypoints_[match.queryIdx].pt,
                FLAGS_loop_history_reprojection_px)) {
            ++rejected_history_reprojection;
            continue;
        }
        used_landmark_ids.insert(map_point->id_);
        points_3d.emplace_back(static_cast<float>(point_world[0]),
                               static_cast<float>(point_world[1]),
                               static_cast<float>(point_world[2]));
        points_2d.push_back(current_frame_->orb_keypoints_[match.trainIdx].pt);
    }

    if (num_correspondences) {
        *num_correspondences = static_cast<int>(points_3d.size());
    }
    if (FLAGS_loop_history_reprojection_px > 0) {
        LOG(INFO) << "Loop historical reprojection rejected="
                  << rejected_history_reprojection;
    }

    if (points_3d.size() < min_pnp_correspondences) {
        if (num_inliers) {
            *num_inliers = 0;
        }
        return false;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_left_->fx_, 0,
                 camera_left_->cx_, 0, camera_left_->fy_, camera_left_->cy_,
                 0, 0, 1);
    cv::Mat rvec, tvec, inliers;
    const bool pnp_success = cv::solvePnPRansac(
        points_3d, points_2d, K, cv::Mat(), rvec, tvec, false, 100,
        FLAGS_loop_pnp_reprojection_px,
        0.99, inliers);

    const int inlier_count = pnp_success ? inliers.rows : 0;
    if (num_inliers) {
        *num_inliers = inlier_count;
    }

    const double inlier_ratio =
        points_3d.empty() ? 0.0
                          : static_cast<double>(inlier_count) /
                                static_cast<double>(points_3d.size());

    if (pnp_success && relative_pose) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Mat33 R;
        Vec3 t;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R(r, c) = R_cv.at<double>(r, c);
            }
            t[r] = tvec.at<double>(r, 0);
        }

        SE3 current_pose_pnp(SO3(R), t);
        *relative_pose = current_pose_pnp * candidate_pose.inverse();
    }

    LOG(INFO) << "PnP verify keyframe " << candidate->keyframe_id_
              << " -> " << current_frame_->keyframe_id_
              << ": correspondences=" << points_3d.size()
              << ", inliers=" << inlier_count
              << ", inlier_ratio=" << inlier_ratio;

    return pnp_success && inlier_count >= min_pnp_inliers &&
           inlier_ratio >= min_inlier_ratio;
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        auto mp = kp->map_point_.lock();
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    ExtractORBForKeyframe();
    AssociateORBWithMapPoints();
    MatchORBWithLastKeyframe();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam
