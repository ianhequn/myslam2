add_executable(run_kitti_stereo run_kitti_stereo.cpp)
target_link_libraries(run_kitti_stereo myslam ${THIRD_PARTY_LIBS})
add_executable(replay_pose_graph replay_pose_graph.cpp)
target_link_libraries(replay_pose_graph myslam ${THIRD_PARTY_LIBS})
