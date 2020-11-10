cmake_minimum_required(VERSION 3.13)
project(coral_pose_detection_cpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(edgetpu_test
#        edgetpu_pose_detection_test.cpp
        edgetpu_working_test.cpp)

include_directories(~/pose_detection/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(~/pose_detection/tensorflow/tensorflow/lite/)
include_directories(~/pose_detection/tensorflow)

target_link_libraries(edgetpu_test /home/mendel/pose_detection/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a)
target_link_libraries(edgetpu_test edgetpu)
target_link_libraries(edgetpu_test pthread)
target_link_libraries(edgetpu_test m)
target_link_libraries(edgetpu_test dl)

###       Copying the source files for the runnable file
add_custom_command(TARGET edgetpu_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/src $<TARGET_FILE_DIR:edgetpu_test>/src)