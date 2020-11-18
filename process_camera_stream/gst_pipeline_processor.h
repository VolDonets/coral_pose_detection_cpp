//
// Created by biba_bo on 2020-08-21.
//

#ifndef IRON_TURTLE_REAR_SIGHT_WEBRTC_MANIPULATION_H
#define IRON_TURTLE_REAR_SIGHT_WEBRTC_MANIPULATION_H

#include <iostream>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#ifdef G_OS_UNIX
#include <glib-unix.h>
#endif

#include "../pose_detector_src/pose_detector_wrapper.h"

#define STR_WIDTH       "640"
#define STR_HEIGHT      "480"
#define STR_FRAMERATE   "15/1"
#define STR_IP          "192.168.1.12"
#define STR_PORT        "5000"
#define STR_AUTO_MULTICAST "FALSE"
const int WIDTH = 640;
const int HEIGHT = 480;

static GstPadProbeReturn cb_have_data(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
void create_and_run_pipeline();
#ifdef G_OS_UNIX
gboolean exit_sighandler (gpointer user_data);
#endif //G_OS_UNIX
int start_gst_loop();

static std::unique_ptr<PoseDetectorWrapper> s_poseDetectorWrapper;

#endif //IRON_TURTLE_REAR_SIGHT_WEBRTC_MANIPULATION_H
