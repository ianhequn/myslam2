//
// Created by gaoxiang on 19-5-4.
//
//本代码是全程序的入口部分
#include <gflags/gflags.h>//来自google gflags的命令行参数

#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");
//生成一个全局字符串的变量
//config_file              参数名称
//./config/default.yaml    默认值
//config file path         帮助说明
int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
