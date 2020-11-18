//
// Created by biba_bo on 2020-08-21.
//


#include "gst_pipeline_processor.h"


/// a GstPad callback function, it is used for modification a pipeline stream
static GstPadProbeReturn cb_have_data(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstMapInfo map;
    GstBuffer *buffer;

    buffer = GST_PAD_PROBE_INFO_BUFFER (info);

    buffer = gst_buffer_make_writable (buffer);

    /* Making a buffer writable can fail (for example if it
     * cannot be copied and is used more than once)
     */
    if (buffer == NULL)
        return GST_PAD_PROBE_OK;

    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        cv::Mat mainImage = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC4, (char *) (map.data), cv::Mat::AUTO_STEP);
        cv::Mat mainImageClone = mainImage.clone();

        s_poseDetectorWrapper->add_frame(mainImageClone);
        s_poseDetectorWrapper->draw_last_pose_on_image(mainImage);

        gst_buffer_unmap(buffer, &map);
    }

    GST_PAD_PROBE_INFO_DATA (info) = buffer;

    return GST_PAD_PROBE_OK;
}

//a function fro filling a ReceiverEntry structure
//Here creates a pipeline, and adds a callback function for stream modifications
void create_and_run_pipeline() {
    std::cout << "Starting a pipeline!" << "\n";
    GError *error;

    error = NULL;

    GstElement *pipeline =
            gst_parse_launch(""
                             "v4l2src device=/dev/video1 "
                             "! video/x-raw,width=" STR_WIDTH ",height=" STR_HEIGHT ",framerate= " STR_FRAMERATE " "
                             "! videoconvert name=ocvvideosrc "
                             "! video/x-raw,format=BGRA "
                             "! videoconvert "
                             "! queue max-size-buffers=1 "
                             "! x264enc speed-preset=ultrafast tune=zerolatency key-int-max=15 "
                             "! video/x-h264,profile=constrained-baseline ! queue max-size-time=0 "
                             "! h264parse "
                             "! rtph264pay config-interval=10 pt=96 "
                             "! udpsink host=" STR_IP " auto-multicast=" STR_AUTO_MULTICAST " port=" STR_PORT "", &error);


    if (error != NULL) {
        g_error("Could not create UDP pipeline: %s\n", error->message);
        g_error_free(error);
        goto cleanup;
    }

    GstElement *ocvvideosrc;
    ocvvideosrc = gst_bin_get_by_name(GST_BIN(pipeline), "ocvvideosrc");

    GstPad *pad;
    pad = gst_element_get_static_pad(ocvvideosrc, "src");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, (GstPadProbeCallback) cb_have_data, NULL, NULL);
    gst_object_unref(pad);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    cleanup:
    gst_object_unref(GST_OBJECT (pipeline));
}

#ifdef G_OS_UNIX
gboolean exit_sighandler (gpointer user_data) {
    g_print ("Caught signal, stopping mainloop\n");
    GMainLoop *mainloop = (GMainLoop *) user_data;
    g_main_loop_quit (mainloop);
    return TRUE;
}
#endif

int start_gst_loop() {
    s_poseDetectorWrapper = std::make_unique<PoseDetectorWrapper>();
    s_poseDetectorWrapper->start_pose_detection();

    setlocale(LC_ALL, "");
    gst_init(nullptr, nullptr);

    GMainLoop *mainloop;
    mainloop = g_main_loop_new(NULL, FALSE);
    g_assert (mainloop != NULL);

#ifdef G_OS_UNIX
    g_unix_signal_add(SIGINT, exit_sighandler, mainloop);
    g_unix_signal_add(SIGTERM, exit_sighandler, mainloop);
#endif

    create_and_run_pipeline();

    g_main_loop_run(mainloop);

    g_main_loop_unref(mainloop);

    return 0;
}
