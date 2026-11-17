//
//
//主控制代码
#include "myslam/visual_odometry.h"
#include <chrono>
#include <iomanip>
#include <map>
#include "myslam/config.h"

namespace myslam {

VisualOdometry::VisualOdometry(std::string &config_path, bool use_viewer)
    : use_viewer_(use_viewer), config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    //创建 Dataset / Frontend / Backend / Map / Viewer，并把它们连起来
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    dataset_ =
        Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    if (!dataset_->Init()) {
        LOG(ERROR) << "Failed to initialize the dataset";
        return false;
    }

    // Open output before starting backend and viewer threads.
    trajectory_file_.open("./results/baseline_kitti_00.txt");
    if (!trajectory_file_) {
        LOG(ERROR) << "cannot open trajectory file; create ./results first";
        return false;
    }

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    if (use_viewer_) viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    if (viewer_) viewer_->SetMap(map_);

    return true;
}
//while 循环一直跑
void VisualOdometry::Run() {
    while (1) {
        LOG(INFO) << "VO is running";
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    SaveKeyframeTrajectory("./results/keyframes_before.txt");
    backend_->OptimizePoseGraph();
    SaveKeyframeTrajectory("./results/keyframes_after.txt");
    if (viewer_) viewer_->Close();

    if (trajectory_file_.is_open()) {
        trajectory_file_.close();
    }

    LOG(INFO) << "VO exit";
}
void VisualOdometry::SaveKeyframeTrajectory(const std::string &path) {
    std::ofstream output(path);
    if (!output) {
        LOG(ERROR) << "cannot open keyframe trajectory: " << path;
        return;
    }
    const auto keyframes = map_->GetAllKeyFrames();
    std::map<unsigned long, Frame::Ptr> ordered(keyframes.begin(), keyframes.end());
    output << std::fixed << std::setprecision(9);
    for (const auto &entry : ordered) {
        const auto &frame = entry.second;
        auto pose = frame->Pose().inverse().matrix();
        output << frame->id_;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col) output << " " << pose(row, col);
        output << '\n';
    }
    LOG(INFO) << "Saved " << ordered.size() << " keyframes to " << path;
}
//读取一帧图像，然后交给 Frontend
bool VisualOdometry::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    if (success) {
        SaveTrajectory(new_frame);
    }
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return success;
}
void VisualOdometry::SaveTrajectory(Frame::Ptr frame) {
    SE3 Twc = frame->Pose().inverse();
    auto T = Twc.matrix();

    trajectory_file_ << std::fixed << std::setprecision(9)
                     << T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " " << T(0, 3) << " "
                     << T(1, 0) << " " << T(1, 1) << " " << T(1, 2) << " " << T(1, 3) << " "
                     << T(2, 0) << " " << T(2, 1) << " " << T(2, 2) << " " << T(2, 3)
                     << std::endl;
}
}  // namespace myslam

