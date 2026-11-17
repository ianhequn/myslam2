1.add_executable(run_kitti_stereo run_kitti_stereo.cpp)
格式是：
add_executable(目标名称 源文件)
这代表着指导编译器编译那个文件把产物放在哪里
其中在根文件的设置中已经确认把所有的编译产物放在bin中
2.target_link_libraries(
    run_kitti_stereo
    myslam
    ${THIRD_PARTY_LIBS}
)
表示的是在生成这个编译产物的适合连接什么算法库或者第三方库
lib/libmyslam.so
它由 src/CMakeLists.txt 创建：
add_library(myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    feature.cpp
    frontend.cpp
    backend.cpp
    viewer.cpp
    visual_odometry.cpp
    dataset.cpp
)