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

    /*if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        cv::Mat done_main_image, done_mini_image;
        cv::Size frame_size(WIDTH, HEIGHT);

        cv::Mat main_image = cv::Mat(frame_size, CV_8UC4, (char *) (map.data), cv::Mat::AUTO_STEP);


        if (count_frames == CHECK_PER_FRAMES) {
            cv::Mat checking_mat = main_image.clone();
            form_detection_processor->add_frame(checking_mat);
            count_frames = 0;
        }
        count_frames++;
        std::vector<cv::Rect> *faces_coord = form_detection_processor->getLastDetectedFaces();
        if (faces_coord != nullptr) {
            // this line draw all detected ROIs, WITHOUT interpolation
            for (int i = 0; i < faces_coord->size(); i++)
                cv::rectangle(main_image, faces_coord->operator[](i), cv::Scalar(0, 255, 0), 2, 0, 0);
        }


        cv::Mat copy_main_image = main_image.clone();

        cv::Rect my_interest_region(frame_param->CROPPED_X, frame_param->CROPPED_Y,
                                    frame_param->CROPPED_WIDTH, frame_param->CROPPED_HEIGHT);
        cv::rectangle(copy_main_image, my_interest_region, cv::Scalar(0, 0, 255), 2, 0, 0);

        cv::Mat cropped_img = main_image(my_interest_region);
        cv::resize(cropped_img, done_main_image, cv::Size(WIDTH, HEIGHT));
        cv::resize(copy_main_image, done_mini_image, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

        cv::Rect main_insertion_coord(0, 0, WIDTH, HEIGHT);
        cv::Rect mini_insertion_coord(RESIZE_X, RESIZE_Y, RESIZE_WIDTH, RESIZE_HEIGHT);

        done_main_image.copyTo(main_image(main_insertion_coord));
        done_mini_image.copyTo(main_image(mini_insertion_coord));

        cv::circle(main_image, cv::Point(DRAW_CIRCLE_X, DRAW_CIRCLE_Y), DRAW_CIRCLE_RADIUS, cv::Scalar(0, 0, 255), 2,
                   cv::LINE_8, 0);
        cv::line(main_image, cv::Point(DRAW_LINE_1B_X, DRAW_LINE_1B_Y), cv::Point(DRAW_LINE_1E_X, DRAW_LINE_1E_Y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        cv::line(main_image, cv::Point(DRAW_LINE_2B_X, DRAW_LINE_2B_Y), cv::Point(DRAW_LINE_2E_X, DRAW_LINE_2E_Y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        cv::line(main_image, cv::Point(DRAW_LINE_3B_X, DRAW_LINE_3B_Y), cv::Point(DRAW_LINE_3E_X, DRAW_LINE_3E_Y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        cv::line(main_image, cv::Point(DRAW_LINE_4B_X, DRAW_LINE_4B_Y), cv::Point(DRAW_LINE_4E_X, DRAW_LINE_4E_Y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_8);

        // printing a speed value (here is a commands for the left and the right wheels)
        cv::putText(main_image, current_speed, cv::Point(5, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);


        std::cout << "Hey, bro!\n";
        gst_buffer_unmap(buffer, &map);
    }*/

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
                             "! udpsink host=" STR_IP " auto-multicast=true port=" STR_PORT "", &error);


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
