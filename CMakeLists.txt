cmake_minimum_required(VERSION 3.13)
project(coral_pose_detection_cpp)

set(CMAKE_CXX_STANDARD 20)

include_directories(/home/mendel/pose_detection/tensorflow_special_commit/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(/home/mendel/pose_detection/tensorflow_special_commit/tensorflow/tensorflow/lite/)
include_directories(/home/mendel/pose_detection/tensorflow_special_commit/tensorflow)

function(dep_edgetpu_tsl EXECUTABLE)
    target_link_libraries(${EXECUTABLE} /home/mendel/pose_detection/libedgetpu-local/libedgetpu/direct/aarch64/libedgetpu.so.1.0)
    target_link_libraries(${EXECUTABLE} /home/mendel/pose_detection/tensorflow_special_commit/tensorflow/tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a)

    target_link_libraries(${EXECUTABLE} pthread)
    target_link_libraries(${EXECUTABLE} m)
    target_link_libraries(${EXECUTABLE} dl)
    target_link_libraries(${EXECUTABLE} rt)

    ###  Adding to the project the 'opencv' library it used for the frame processing (and, +- as an object detector) and as an object tracker
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE} ${OpenCV_LIBS})
endfunction()

add_executable(pose_stream_test
        posenet/posenet_decoder.h
        posenet/posenet_decoder.cc
        posenet/posenet_decoder_op.h
        posenet/posenet_decoder_op.cc

        pose_detector_src/pose_detector_wrapper.h
        pose_detector_src/pose_detector_wrapper.cpp

        process_camera_stream/gst_pipeline_processor.h
        process_camera_stream/gst_pipeline_processor.cpp

        main.cpp)
dep_edgetpu_tsl(pose_stream_test)
target_compile_definitions(pose_stream_test PRIVATE MY_DEBUG_DEF)

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(GST "gstreamer-1.0")
    if (GST_FOUND)
        target_link_libraries(pose_stream_test ${GST_LIBRARIES})
        add_definitions(${GST_CFLAGS} ${GST_CFLAGS_OTHER})
    endif ()
endif ()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(SPD "gstreamer-sdp-1.0")
    if (SPD_FOUND)
        target_link_libraries(pose_stream_test ${SPD_LIBRARIES})
        add_definitions(${SPD_CFLAGS} ${SPD_CFLAGS_OTHER})
    endif ()
endif ()


#add_executable(edgetpu_test
#        posenet/posenet_decoder.h
#        posenet/posenet_decoder.cc
#        posenet/posenet_decoder_op.h
#        posenet/posenet_decoder_op.cc
#
#        edgetpu_pose_detection_test.cpp)
#
#target_compile_definitions(edgetpu_test PRIVATE MY_DEBUG_DEF)
#dep_edgetpu_tsl(edgetpu_test)

####       Copying the source files for the runnable file
#add_custom_command(TARGET edgetpu_test POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory
#        ${PROJECT_SOURCE_DIR}/src $<TARGET_FILE_DIR:edgetpu_test>/src)

#add_executable(pose_detector_wrapper_tester
#        posenet/posenet_decoder.h
#        posenet/posenet_decoder.cc
#        posenet/posenet_decoder_op.h
#        posenet/posenet_decoder_op.cc
#
#        pose_detector_src/pose_detector_wrapper.h
#        pose_detector_src/pose_detector_wrapper.cpp
#
#        pose_detector_wrapper_tester.cpp)
#
#dep_edgetpu_tsl(pose_detector_wrapper_tester)